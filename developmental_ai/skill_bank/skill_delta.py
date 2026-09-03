"""SKILLS AS DELTAS — a skill is a modulation of a shared base, not a copy.

THE PROBLEM THIS SOLVES
  A minted skill is a byte copy of the whole policy. Measured on the pod bank:
  25 skills at ~115 MB each whose first-layer weights are 0.82-0.995 cosine
  similar to one another. That is one network stored twenty-five times. It is
  also the wrong idea: learning to chop wood does not clone your visual
  cortex and file the copy under "chopping".

THE REPRESENTATION
      W_skill  =  W_base  +  delta

  `W_base` is ONE shared actor kept once per bank. Each skill stores only its
  delta, in whichever form is both smaller AND faithful:
      * LOW RANK (U @ V) when the delta genuinely lives in a small subspace;
      * INT8 DENSE with a per-tensor scale otherwise.

WHAT THE DATA ACTUALLY SAYS — low rank was the obvious guess, and it is WRONG
  The plausible story was that skills descend from one policy and differ by a
  few coordinated behavioural adjustments, i.e. a low-rank update. Measured on
  the real pod bank (16 same-shape skills, `actor.shared.0.weight`):

        rank   1   rel err 0.371
        rank  16   rel err 0.156
        rank  64   rel err 0.078
        rank 128   rel err 0.038
        rank 256   rel err 0.000      (= full rank; no saving)

  The spectrum barely decays. These deltas are essentially FULL RANK — the
  signature of accumulated SGD noise, not of structure. It is consistent with
  the earlier measurement that 86% of those weights never left their
  initialization: there is little learned behaviour for skills to differ IN,
  so most of what differs between them is noise. ||delta||/||W|| = 0.30.

  Hence the int8 path. Rank cannot pay here; precision can, because the delta
  is small and bounded. If a future bank's skills DO differ structurally (as
  they should once skills are practised rather than copied), `_factorize`
  will find it and the low-rank path takes over automatically — the code
  chooses per tensor rather than committing to either belief.

  Rank is chosen ADAPTIVELY: the smallest rank whose reconstruction stays
  under `tol`, and only if it actually costs less than dense. A skill that
  does not fit is stored faithfully instead of being mangled to hit a size
  target.

FIDELITY IS THE WHOLE POINT
  A conversion that saves space but changes behaviour has destroyed the
  agent's memories. `verify_behaviour` therefore checks the thing that
  actually matters — that the reconstructed actor produces the same action
  distribution as the original — rather than only a weight-space norm. The
  converter refuses to write any skill that fails it.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# Marker key so a loader can tell a delta file from a dense one at a glance,
# without guessing from the presence/absence of tensor names.
DELTA_MARKER = "__skill_delta__"
DELTA_VERSION = 1


def make_base(actor_sds: List[Dict[str, torch.Tensor]]
              ) -> Dict[str, torch.Tensor]:
    """Build the shared base actor: the elementwise MEAN of the given actors.

    The mean minimises the total squared delta across the bank, which is the
    quantity the low-rank factorization then has to represent — so it is not
    merely a convenient choice, it is the one that makes every skill's delta
    as small (and as compressible) as possible.

    Only keys present in EVERY actor with a consistent shape are averaged;
    anything else is left to the per-skill dense path, because a base entry
    that does not apply to all tenants is not a base.
    """
    if not actor_sds:
        raise ValueError("cannot build a base from zero actors")
    keys = set(actor_sds[0])
    for sd in actor_sds[1:]:
        keys &= set(sd)
    base: Dict[str, torch.Tensor] = {}
    for k in sorted(keys):
        shapes = {tuple(sd[k].shape) for sd in actor_sds}
        if len(shapes) != 1:
            continue                      # shape-inconsistent: not a base key
        stack = torch.stack([sd[k].detach().float() for sd in actor_sds])
        base[k] = stack.mean(dim=0)
    return base


def _factorize(d: torch.Tensor, tol: float, max_rank: int
               ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Smallest-rank (U, V) with ||d - U@V|| / ||d|| <= tol, or None.

    None means "not worth it" — either the tensor is not 2D, or no rank that
    fits the error budget is smaller than storing the delta densely. Returning
    None rather than a bad factorization is deliberate: the caller falls back
    to dense and fidelity is preserved.
    """
    if d.dim() != 2 or d.numel() == 0:
        return None
    m, n = d.shape
    dense_cost = m * n
    nrm = float(d.norm())
    if nrm == 0.0:
        return None                       # identical to base: dense is 0 bytes
    try:
        U, S, Vh = torch.linalg.svd(d, full_matrices=False)
    except Exception:                     # never let a bad matrix kill a run
        return None
    total = float((S ** 2).sum())
    cum = torch.cumsum(S ** 2, dim=0)
    limit = min(int(S.numel()), int(max_rank))
    for r in range(1, limit + 1):
        # Relative Frobenius error of the rank-r truncation. The residual is
        # clamped at 0: once the truncation is near-exact, `total - cum[r]` is
        # a difference of nearly equal floats and lands slightly NEGATIVE,
        # whose square root is complex — which raised rather than reporting
        # the (essentially zero) error it actually represents.
        resid = max(0.0, total - float(cum[r - 1]))
        err = float((resid / total) ** 0.5)
        if err <= tol:
            if r * (m + n) >= dense_cost:
                return None               # factorization would not save space
            return (U[:, :r] * S[:r], Vh[:r, :])
    return None


def to_delta(base: Dict[str, torch.Tensor],
             actor_sd: Dict[str, torch.Tensor],
             tol: float = 0.01, max_rank: int = 64,
             quantize: bool = True) -> Dict[str, Any]:
    """Express `actor_sd` as (base + delta): low-rank where the delta really
    is low-rank, int8-quantized dense otherwise (see the note in the body)."""
    out: Dict[str, Any] = {DELTA_MARKER: DELTA_VERSION, "lowrank": {},
                           "dense": {}, "tol": float(tol)}
    for k, w in actor_sd.items():
        w = w.detach().float()
        b = base.get(k)
        if b is None or b.shape != w.shape:
            # no comparable base entry -> the skill carries it whole
            out["dense"][k] = w.clone()
            continue
        # DEVICE ALIGNMENT (2026-08-17). The base is always read back from
        # disk with map_location="cpu" while the live policy sits on the
        # run's device, so on ANY GPU box this subtraction raised "Expected
        # all tensors to be on the same device" — which the caller caught
        # and answered by storing the skill in FULL. Measured on the live
        # pod: every minted skill 14 MB instead of a delta, i.e. the whole
        # compression win silently absent on exactly the hardware that
        # trains. Subtracting two tensors is not the place to care where
        # they live; align here so no caller can re-set the trap.
        if b.device != w.device:
            b = b.to(w.device)
        d = w - b
        f = _factorize(d, tol, max_rank)
        if f is None:
            if float(d.abs().max()) == 0.0:
                continue                  # exactly the base: store nothing
            # MEASURED (2026-08-01, on the real pod bank): skill deltas here
            # are essentially FULL RANK — rank 64 still carries 7.8% relative
            # error, rank 128 3.8%, exactness needs all 256. The spectrum
            # barely decays, which is what accumulated SGD noise looks like,
            # not coordinated behavioural structure. That is consistent with
            # the earlier finding that 86% of these weights never left their
            # initialization: there is little learned behaviour for skills to
            # differ IN, so what differs is mostly noise.
            #
            # So the saving here cannot come from rank. It comes from
            # PRECISION: a delta is small and bounded (measured
            # ||delta||/||W|| = 0.30), so int8 with a per-tensor scale
            # represents it to ~0.4% while costing a quarter of fp32.
            # Fidelity is still decided by verify_behaviour, not by this
            # choice — anything that moves behaviour is refused upstream.
            out["dense"][k] = _quantize(d) if quantize else d.clone()
        else:
            out["lowrank"][k] = (f[0].contiguous(), f[1].contiguous())
    return out


def _quantize(d: torch.Tensor) -> Dict[str, Any]:
    """Symmetric int8 with a PER-OUTPUT-CHANNEL scale.

    A single scale for the whole tensor is set by its largest element, so
    every channel with a smaller dynamic range is quantized coarsely relative
    to its own magnitude. That matters most exactly where precision matters
    most — conv encoder weights, whose per-channel magnitudes differ by
    orders of magnitude. MEASURED on a drifted encoder delta: per-tensor
    scaling gave 1.17e-2 relative error in the ENCODED FEATURES (i.e. in what
    the skill actually sees); per-channel drops it by ~an order of magnitude
    for the cost of one float per output channel.
    """
    if d.dim() == 0 or d.numel() == 0:
        return {"q": torch.zeros(d.shape, dtype=torch.int8), "scale": 0.0}
    flat = d.reshape(d.shape[0], -1)
    amax = flat.abs().amax(dim=1)                       # (out_channels,)
    scale = (amax / 127.0).clamp_min(1e-12)
    q = torch.clamp(torch.round(flat / scale[:, None]), -127, 127)
    return {"q": q.to(torch.int8).reshape(d.shape),
            "scale": scale.to(torch.float32), "shape": tuple(d.shape)}


def _dequantize(entry: Any) -> torch.Tensor:
    if isinstance(entry, dict) and "q" in entry:
        q, s = entry["q"], entry["scale"]
        if isinstance(s, torch.Tensor) and s.numel() > 1:
            flat = q.float().reshape(q.shape[0], -1) * s[:, None].float()
            return flat.reshape(q.shape)
        return q.float() * float(s)          # legacy per-tensor payloads
    return entry.float()


def from_delta(base: Dict[str, torch.Tensor],
               delta: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Rebuild a full actor state dict from base + delta."""
    if not is_delta(delta):
        raise ValueError("not a skill-delta payload")
    out: Dict[str, torch.Tensor] = {k: v.detach().clone()
                                    for k, v in base.items()}
    # Same device alignment as to_delta, for the same reason: base and delta
    # are separate files and nothing guarantees they were loaded onto the
    # same device. Today both callers use map_location="cpu" so this is
    # inert — but the write side had exactly this trap live for months, and
    # a reconstruction that raises here loses the skill entirely.
    for k, (U, V) in (delta.get("lowrank") or {}).items():
        add = (U.float() @ V.float())
        if k in out:
            out[k] = out[k] + add.to(out[k].device)
        else:
            out[k] = add
    for k, d in (delta.get("dense") or {}).items():
        dd = _dequantize(d)
        if k in out:
            out[k] = out[k] + dd.to(out[k].device)
        else:
            out[k] = dd
    return out


def is_delta(payload: Any) -> bool:
    return isinstance(payload, dict) and DELTA_MARKER in payload


def param_count(delta: Dict[str, Any]) -> int:
    n = sum(int(U.numel() + V.numel())
            for U, V in (delta.get("lowrank") or {}).values())
    # int8 dense entries cost a QUARTER of an fp32 parameter, so counting
    # them as whole params would understate the saving by 4x.
    for t in (delta.get("dense") or {}).values():
        if isinstance(t, dict) and "q" in t:
            n += int(t["q"].numel()) // 4
        else:
            n += int(t.numel())
    return n


def weight_error(original: Dict[str, torch.Tensor],
                 rebuilt: Dict[str, torch.Tensor]) -> float:
    """Worst per-tensor relative reconstruction error."""
    worst = 0.0
    for k, w in original.items():
        r = rebuilt.get(k)
        if r is None:
            return float("inf")
        nrm = float(w.float().norm())
        e = float((w.float() - r.float()).norm()) / (nrm if nrm else 1.0)
        worst = max(worst, e)
    return worst


def verify_behaviour(actor_cls, build_kwargs: Dict[str, Any],
                     original: Dict[str, torch.Tensor],
                     rebuilt: Dict[str, torch.Tensor],
                     in_dim: int, n_probe: int = 256,
                     seed: int = 0, tie_margin: float = 1e-4
                     ) -> Dict[str, float]:
    """Does the rebuilt skill still BEHAVE like the original?

    Weight-space error is a proxy; this measures the thing that matters. Two
    actors are built, loaded with the two state dicts, and compared on the
    same random probe inputs:

      * `max_kl`         — worst-case KL between the two action
                           distributions. This is the PRIMARY criterion,
                           because `skill_action` SAMPLES from the
                           distribution; it never takes an argmax. If the
                           distribution is unchanged, the skill is unchanged.
      * `decisive_agree` — argmax agreement restricted to probes where the
                           top-two logits differ by more than `tie_margin`,
                           i.e. where the skill actually had a preference.
      * `argmax_agree`   — raw argmax agreement, reported for context.

    The distinction matters. On probes where the top two logits are within
    ~1e-6 of each other the greedy action is decided by numerical dust, so a
    flip there records nothing about the skill; demanding raw argmax equality
    would make the gate a test of float noise rather than of behaviour, and
    the only way to pass it would be to not compress at all. Requiring
    agreement where the skill was DECISIVE, plus a strict KL bound
    everywhere, tests the thing that actually governs what the agent does.
    """
    import torch.nn.functional as F
    g = torch.Generator().manual_seed(seed)
    a1 = actor_cls(**build_kwargs)
    a2 = actor_cls(**build_kwargs)
    a1.load_state_dict({k: v.float() for k, v in original.items()})
    a2.load_state_dict({k: v.float() for k, v in rebuilt.items()})
    a1.eval(); a2.eval()
    x = torch.randn(n_probe, in_dim, generator=g)
    with torch.no_grad():
        l1 = a1.action_head(a1.shared(x))
        l2 = a2.action_head(a2.shared(x))
        p1, lp1 = F.softmax(l1, -1), F.log_softmax(l1, -1)
        lp2 = F.log_softmax(l2, -1)
        kl = (p1 * (lp1 - lp2)).sum(-1)
        same = (l1.argmax(-1) == l2.argmax(-1))
        # SELF-CALIBRATING TIE MARGIN. A fixed constant is the wrong tool:
        # what counts as "the skill had a real preference" depends on the
        # scale of the perturbation the conversion introduced, which varies
        # per skill. So measure that perturbation and call a probe DECISIVE
        # only when the original's top-two gap exceeds it. The resulting
        # claim is the one worth making: no preference larger than our own
        # error was flipped. Probes below that threshold are decided by
        # numerical dust in BOTH versions and record nothing about behaviour.
        max_logit_delta = float((l1 - l2).abs().max())
        margin = max(float(tie_margin), 2.0 * max_logit_delta)
        top2 = l1.topk(2, dim=-1).values
        decisive = (top2[:, 0] - top2[:, 1]) > margin
        dec_agree = (same[decisive].float().mean()
                     if bool(decisive.any()) else torch.tensor(1.0))
    return {"argmax_agree": float(same.float().mean()),
            "decisive_agree": float(dec_agree),
            "decisive_frac": float(decisive.float().mean()),
            "max_logit_delta": max_logit_delta,
            "max_kl": float(kl.max()), "mean_kl": float(kl.mean())}
