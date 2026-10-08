"""
Learning-Progress curiosity — the Rung 3 alternative to raw-novelty (ICM).

Raw-novelty curiosity rewards prediction ERROR, so an uncontrollable noise source
(the "noisy TV") pays out forever and traps the agent. Learning-progress (LP)
curiosity instead rewards the REDUCTION of prediction error over repeated visits
to a state — i.e. *getting better at predicting*, which is what actually signals
masterable structure (Oudeyer/IMGEP). On pure noise you never get better, so
LP → 0 there; in learnable regions error falls with experience, so LP > 0 until
mastered (then → 0, and the agent moves on).

DESIGN — a controlled swap, not a different model.
This subclasses IntrinsicCuriosityModule and overrides ONLY
`compute_intrinsic_reward`. It reuses the exact same feature encoder + forward
model + `train_step`, so the per-transition prediction error is computed
identically to the novelty arm; the *only* difference between the two Rung-3 arms
is novelty (raw error) vs LP (error reduction). That makes the comparison clean.

LP per transition:
  * bucket the transition by a coarse hash of its observation;
  * keep a short history of this bucket's recent forward-model errors, ONE
    entry per VISIT (consecutive same-bucket steps of a stream collapse to
    their mean; LP is scored once, on entry — see `lp_visit_collapse`);
  * LP = max(0, mean(older half) − mean(recent half))  — the drop in error,
    paid only if it clears a significance test that stays valid when the
    history is autocorrelated (see `lp_sig_z`; the 2026-10 micro-turn farm).
Crucially, the noisy TV corrupts the observation every step, so each TV
transition lands in a NEW (singleton) bucket → no history → LP = 0. Learnable
states recur → their buckets accumulate → error falls → LP > 0. The asymmetry
(noise = always singleton, learnable = recurs) is what makes LP ignore the TV.
"""

from __future__ import annotations

from collections import deque
import math

import numpy as np
import torch
import torch.nn.functional as F

from developmental_ai.curiosity.icm import IntrinsicCuriosityModule


class LearningProgressCuriosity(IntrinsicCuriosityModule):
    """ICM whose intrinsic reward is LEARNING PROGRESS (error reduction per state
    bucket) rather than raw prediction error. Drop-in for IntrinsicCuriosityModule."""

    def __init__(self, *args, lp_history: int = 30, lp_min_samples: int = 4,
                 lp_bucket_round: int = 1, lp_max_buckets: int = 100_000,
                 lp_pixel_pool: int = 4, lp_proto_thresh: float = 0.025,
                 lp_max_protos: int = 512, lp_abs_frac: float = 0.0,
                 action_conditional: bool = False, null_action: int = 0,
                 flow_error_weight: float = 1.0,
                 lp_sig_z: float = 2.33, lp_rho_max: float = 0.9,
                 lp_visit_collapse: bool = True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # ---- A SIGNIFICANCE TEST THAT HOLDS UNDER AUTOCORRELATION ---------
        # ---- (2026-10, the micro-turn farm) -------------------------------
        # The gate below used to be `drop > 0.5 * std(history)`. That is a
        # test for INDEPENDENT samples, and the history is not independent:
        # the smallest non-noop actions (2-degree micro-turn = action 17,
        # 120-tick attack hold = action 27) leave the 4x4-pooled frame almost
        # unchanged, so the agent sits in ONE prototype for many steps and
        # the 30-entry history fills with back-to-back errors of a model that
        # has not changed. A correlated series has a large half-mean
        # difference relative to its own std, so chance passed the gate:
        #     frozen model, iid errors          ->  9% of transitions paid
        #     frozen model, AR(1) rho 0.5-0.95  -> 22-39% paid
        #     LIVE (13 h shadow): action 17 passed 24-28%, action 27 27%,
        #     everything else ~7%, and action 17's LP pay GREW
        #     0.03 -> 0.26/step — learning progress fabricated from nothing.
        # Noop pays exactly 0 (action-attribution gate), so the farm settled
        # on the smallest action that is not noop.
        #
        # TWO CHANGES, both needed (tests/_lp_correlated_gate_smoke.py):
        #  1. VISIT COLLAPSE (lp_visit_collapse). Consecutive same-bucket
        #     transitions of one batch row (= one stream) are ONE visit: the
        #     history gets ONE entry, the visit's mean error, when the visit
        #     ends; LP is scored ONCE, on the entry transition. A dwell can
        #     no longer be paid N times for one piece of evidence, and the
        #     history means what the module docstring always said it meant —
        #     error over repeated VISITS. Escape path: any bucket change
        #     closes the visit; the agent controls that by moving.
        #  2. A VALID TEST (lp_sig_z). Collapse alone cannot help rapid A/B
        #     oscillation (every step is an entry, entries 2 steps apart).
        #     So `drop` is compared with the standard error of the difference
        #     of the two half-means — pooled WITHIN-half variance, so a real
        #     drop does not inflate its own noise estimate — inflated by
        #     (1+r)/(1-r) for the history's lag-1 autocorrelation r (the
        #     effective-sample-size correction n_eff = n(1-r)/(1+r)), against
        #     a one-sided ~1% quantile at the effective degrees of freedom.
        #     r is estimated from first differences WITHIN each half
        #     (von Neumann), which a learning trend barely moves, corrected
        #     for its small-sample downward bias, and is
        #     clipped to [0, lp_rho_max] so the threshold stays finite: a
        #     bucket with an extremely correlated history is HARD to pay,
        #     never impossible, and the 30-entry window rolls it over.
        # Revert: lp_sig_z=0 restores the legacy `lp_sig_k * std` gate and
        # lp_visit_collapse=False the per-step history; the smoke keeps the
        # legacy pair as a regression witness that must still farm.
        if not math.isfinite(float(lp_sig_z)) or float(lp_sig_z) < 0:
            raise ValueError("lp_sig_z must be finite and >= 0")
        if not math.isfinite(float(lp_rho_max)) or not 0 <= float(lp_rho_max) < 1:
            raise ValueError("lp_rho_max must be finite and in [0, 1)")
        if int(lp_history) < 2 or not 2 <= int(lp_min_samples) <= int(lp_history):
            raise ValueError("require 2 <= lp_min_samples <= lp_history")
        self.lp_sig_z = float(lp_sig_z)
        self.lp_rho_max = float(lp_rho_max)
        self.lp_visit_collapse = bool(lp_visit_collapse)
        # row index -> [bucket_key, error_sum, n_steps] of the open visit.
        self._lp_runs: dict = {}
        # Relative weight of the world model's flow residual against the
        # forward-model error when the two are summed for bucketing. 1.0 is
        # "both channels count the same"; 0.0 disables the channel entirely
        # and is the revert, without needing the world model rebuilt.
        self.flow_error_weight = float(flow_error_weight)
        # Last flow residual seen, for the telemetry split. 0.0 until a flow
        # head actually supplies one, which is also what it reads in every
        # env that has none.
        self.last_flow_error = 0.0
        # ---- ABSOLUTE PROGRESS FLOOR (2026-08-02) ------------------------
        # LP is a DERIVATIVE, so a world the model has mastered should pay
        # exactly nothing. It did not. The significance gate below is purely
        # RELATIVE (drop > k * the bucket own noise), so on a flat, mastered
        # bucket it still passes a share of pure jitter by construction — and
        # the normalizer then divides by max(running_std, 1e-3) and re-inflates
        # that jitter into a real-looking reward.
        #
        # MEASURED LIVE on a standing-still agent: running_std 1.42e-05, i.e.
        # 70x UNDER the 1e-3 floor, raw prediction error 2.64e-03 — and the
        # base term paid +0.1321/step, 95% of the entire intrinsic drive, for
        # doing nothing. Standing still was the most profitable thing
        # available and no shaping term could outbid it.
        #
        # This requires the error drop to ALSO be a meaningful FRACTION of the
        # bucket own current error. Unit-free and bucket-local, like the
        # existing gate. Real learning drops are large relative to the error;
        # jitter on a mastered bucket is not. 0.0 = off = byte-identical.
        self.lp_abs_frac = float(lp_abs_frac)
        # ---- ACTION-CONDITIONAL CURIOSITY (2026-08-04) -------------------
        # MEASURED: the policy converged deterministically on `noop` (80% of
        # actions, entropy 0.00) while still collecting +0.2951/step. Doing
        # nothing paid almost as well as acting, so a converged policy
        # correctly chose it.
        #
        # WHY it paid: intrinsic reward is forward-model prediction ERROR, and
        # on a live multiplayer server the world changes without the agent —
        # day/night cycles, mobs, other players, foliage. The env asks for
        # frozen time via TimeInitialCondition(allow_passage_of_time=False),
        # but that is a SERVER-side handler and a foreign Paper server ignores
        # it (confirmed: POV frames from one run show both daylight and
        # night). So ambient change generated surprise every step regardless
        # of what the agent did — a standing income for existing.
        #
        # THE GATE, not the signal. Learning progress is still measured on the
        # true forward error; what this scales is how much of that transition
        # the agent's ACTION can account for:
        #     err_a    = ||f(phi(s), a)      - phi(s')||^2   (what happened)
        #     err_null = ||f(phi(s), a_null) - phi(s')||^2   (had I done nothing)
        #     gate     = relu(err_null - err_a) / (err_null + eps)   in [0,1]
        #   ambient change  -> both equally wrong -> gate ~ 0
        #   action caused it-> err_a << err_null  -> gate -> 1
        #   THE AGENT DID NOTHING -> identical inputs -> gate EXACTLY 0
        # That last property is the point: noop can no longer earn anything,
        # by construction rather than by tuning.
        #
        # This gates REWARD, never perception: the world model and the ICM
        # still train on every frame, so learning to predict mobs, weather and
        # other players is unaffected — the agent simply stops being PAID for
        # the world moving on its own.
        self.action_conditional = bool(action_conditional)
        self.null_action = int(null_action)
        self.last_action_attribution = 0.0
        self.lp_history = int(lp_history)
        self.lp_min_samples = int(lp_min_samples)
        self.lp_bucket_round = int(lp_bucket_round)
        self.lp_max_buckets = int(lp_max_buckets)
        # PIXEL bucketing (July 2026, Minecraft curiosity redesign): rounding
        # a 49k-dim float frame makes EVERY frame a singleton bucket, so LP
        # would be permanently zero on pixels; and quantized-hash bucketing is
        # boundary-fragile (a crack-sized overlay flips >=1 of the pooled
        # cells' bins almost surely — measured 0/40 stable). Instead:
        # NEAREST-PROTOTYPE bucketing — a tiny online vector quantizer over
        # block-mean-pooled frames (C*pool*pool dims). A frame joins the
        # nearest prototype if the mean-abs distance is under lp_proto_thresh,
        # else becomes a new prototype (LRU-evicted at lp_max_protos, history
        # dropped with it). Crack overlays move the pooled vector by ~0.01 —
        # DETERMINISTICALLY inside the threshold — so all crack stages of a
        # scene share one bucket, and "getting better at predicting breaking"
        # is exactly the error-drop LP measures there. Distinct scenes differ
        # by >>thresh -> separate buckets. Unlearnable noise either lands in
        # singleton buckets (no history) or merges into one bucket whose
        # error never declines — LP = 0 either way (noisy-TV immunity).
        self.lp_pixel_pool = int(lp_pixel_pool)
        self.lp_proto_thresh = float(lp_proto_thresh)
        self.lp_max_protos = int(lp_max_protos)
        self._protos: dict[int, np.ndarray] = {}   # proto_id -> pooled vec
        self._proto_used: dict[int, int] = {}      # proto_id -> last tick
        self._proto_next_id = 0
        self._proto_tick = 0
        self._bucket_err: dict[int, deque] = {}
        # ---- PROTOTYPE MATRIX CACHE (2026-09-02) -------------------------
        # `_bucket_key` used to rebuild `np.stack([...])` over the WHOLE
        # prototype set on every call — once per env step per stream. At
        # lp_max_protos 512 that is cheap; the cap is now 4096, which would
        # make it 8x the allocate-and-copy on a loop whose per-step CPU is
        # already the measured bottleneck (67.3 ms/step). Cached instead,
        # and invalidated by the ONLY TWO WRITERS to `self._protos`:
        # the insert and the LRU eviction, both in `_bucket_key`.
        # A stale matrix would silently match the wrong bucket and read as
        # "LP got worse" with nothing pointing at the cause — which is why
        # both invalidation sites are named here and asserted in
        # tests/_capacity_wave_smoke.py rather than left to review.
        self._proto_arr: "np.ndarray | None" = None
        self._proto_ids: "list | None" = None
        # ---- GATE TELEMETRY (2026-10-07, measurement only) ---------------
        # Counts of what the LP gates did on LIVE (update_state=True) calls
        # since the last pop_gate_stats(). Read-only calls (imagination) are
        # not counted. Nothing reads these back into the reward.
        self._reset_gate_stats()
        self._lp_gate_stage = 0

    def _pooled_vec(self, obs_row: np.ndarray) -> "np.ndarray | None":
        c = int(getattr(self, "image_channels", 3))
        side = int(getattr(self, "image_size", 64))
        p = self.lp_pixel_pool
        try:
            img = obs_row.reshape(c, side, side)
            k = side // p
            return img[:, : k * p, : k * p].reshape(
                c, p, k, p, k).mean(axis=(2, 4)).flatten()
        except Exception:
            return None  # malformed frame

    def _bucket_key(self, obs_row: np.ndarray, update_state: bool = True) -> int:
        if getattr(self, "pixel_obs", False):
            v = self._pooled_vec(obs_row)
            if v is not None:
                if self._protos:
                    # CACHED (see _proto_arr in __init__): rebuilt only when
                    # a prototype was inserted or evicted since the last
                    # call, not once per env step.
                    if self._proto_arr is None:
                        self._proto_ids = list(self._protos.keys())
                        self._proto_arr = np.stack(
                            [self._protos[i] for i in self._proto_ids])
                    ids = self._proto_ids
                    d = np.abs(self._proto_arr - v).mean(axis=1)
                    j = int(d.argmin())
                    if d[j] < self.lp_proto_thresh:
                        if update_state:
                            # `_proto_used` is LRU bookkeeping only — it does
                            # not change the MATRIX, so the cache survives a
                            # hit. Only membership changes invalidate.
                            self._proto_tick += 1
                            self._proto_used[ids[j]] = self._proto_tick
                        return ids[j]
                if not update_state:
                    return -1   # read-only miss: no history -> LP 0, no mint
                # new prototype (evict least-recently-used + its history)
                self._proto_tick += 1
                if len(self._protos) >= self.lp_max_protos:
                    old = min(self._proto_used, key=self._proto_used.get)
                    self._protos.pop(old, None)
                    self._proto_used.pop(old, None)
                    # THE COST OF THE CAP: evicting a prototype throws away
                    # its error history, so a scene the agent had already
                    # mastered comes back reading as brand-new. That is why
                    # lp_max_protos was raised — this line is the mechanism.
                    self._bucket_err.pop(old, None)
                    self._proto_arr = self._proto_ids = None   # WRITER 1
                pid = self._proto_next_id
                self._proto_next_id += 1
                self._protos[pid] = v.astype(np.float32)
                self._proto_used[pid] = self._proto_tick
                self._proto_arr = self._proto_ids = None       # WRITER 2
                return pid
        # VECTOR path: coarse discretization so a recurring state hashes to a
        # stable bucket, while noise-corrupted observations hash uniquely.
        return hash(np.round(obs_row, self.lp_bucket_round).tobytes())

    def _prune_buckets(self):
        # Reclaim memory from the flood of singleton (noise) buckets; keep the
        # informative ones that have accumulated history.
        if len(self._bucket_err) <= self.lp_max_buckets:
            return
        self._bucket_err = {k: v for k, v in self._bucket_err.items() if len(v) > 1}

    def end_visits(self, stream_ids=None, *, valid=True):
        """Close the open visits of `stream_ids` (all streams when None).

        valid=True folds each visit's mean error into its bucket history.
        On an environment restart the CALLER excludes the splice row (it
        spans two worlds) and closes the pre-crash visit as valid — those
        transitions were real. valid=False discards the open visit, for
        callers whose visit is not training evidence (evaluation). A new
        process starts with no open visits: the world is not checkpointed.
        """
        ids = list(self._lp_runs) if stream_ids is None else list(stream_ids)
        for stream in ids:
            run = self._lp_runs.pop(stream, None)
            if run is not None and valid:
                self._close_visit(run)

    def visit_state_dict(self):
        """Separate from neural weights, so old curiosity.pt remains loadable."""
        return {"schema": 1, "histories": dict(self._bucket_err),
                "protos": self._protos, "proto_used": self._proto_used,
                "proto_next": self._proto_next_id, "proto_tick": self._proto_tick,
                "reward_history": list(self.reward_history),
                "normalization": (self._running_mean, self._running_median,
                                  self._running_std)}

    def load_visit_state_dict(self, state):
        """Validate EVERYTHING, then apply. A rejected state raises before
        any field is touched, so the caller's cold start really is the
        freshly-constructed module (see load_pickle_sidecar). The prototype
        size check matters most: a changed lp_pixel_pool / image_channels
        would otherwise load vectors `_bucket_key` cannot compare with, and
        the first np.abs(arr - v) after boot would crash the loop."""
        if not isinstance(state, dict) or state.get("schema") != 1:
            raise ValueError("unsupported curiosity visit state")
        pixel = bool(getattr(self, "pixel_obs", False))
        expected = int(getattr(self, "image_channels", 3)) * self.lp_pixel_pool ** 2
        protos = {}
        for pid, vec in dict(state["protos"]).items():
            v = np.asarray(vec, dtype=np.float32)
            if v.ndim != 1 or (pixel and v.shape[0] != expected):
                raise ValueError(
                    "prototype %r has shape %s, config expects (%d,)"
                    % (pid, tuple(v.shape), expected))
            if not np.all(np.isfinite(v)):
                raise ValueError("prototype %r is not finite" % (pid,))
            protos[int(pid)] = v
        if protos and not pixel:
            raise ValueError("prototypes saved but this module is not pixel_obs")
        used = {int(k): int(t) for k, t in dict(state["proto_used"]).items()}
        if set(used) != set(protos):
            raise ValueError("prototype LRU table does not match prototypes")
        proto_next = int(state["proto_next"])
        if protos and proto_next <= max(protos):
            raise ValueError("prototype id counter would reissue a live id")
        proto_tick = int(state["proto_tick"])
        histories = {}
        for k, h in dict(state["histories"]).items():
            vals = [float(x) for x in h]
            if not all(math.isfinite(x) for x in vals):
                raise ValueError("bucket history %r is not finite" % (k,))
            histories[k] = deque(vals, maxlen=self.lp_history)
        rewards = [float(x) for x in state["reward_history"]]
        norm = tuple(float(x) for x in state["normalization"])
        if len(norm) != 3 or not all(math.isfinite(x) for x in norm + tuple(rewards)):
            raise ValueError("curiosity normalisation state is not finite")
        # Config wins on capacity: evict least-recently-used beyond the cap,
        # history dropped with it (the same rule as live eviction).
        while len(protos) > self.lp_max_protos:
            old = min(used, key=used.get)
            protos.pop(old); used.pop(old); histories.pop(old, None)
        # --- validated; apply ---
        self._bucket_err = histories
        self._protos, self._proto_used = protos, used
        self._proto_next_id, self._proto_tick = proto_next, proto_tick
        self.reward_history.clear()
        self.reward_history.extend(rewards)
        self._running_mean, self._running_median, self._running_std = norm
        self._lp_runs.clear()  # Environment state is not checkpointed.
        self._proto_arr = self._proto_ids = None

    def _close_visit(self, run) -> None:
        """Append a finished visit's MEAN error to its bucket's history.

        A bucket whose prototype was LRU-evicted mid-visit has lost its
        history on purpose (see `_bucket_key`); re-creating it here would
        leak an orphan history for an id that no longer exists."""
        key, err_sum, n = run
        if getattr(self, "pixel_obs", False) and self._protos \
                and key not in self._protos:
            return
        hist = self._bucket_err.get(key)
        if hist is None:
            hist = deque(maxlen=self.lp_history)
            self._bucket_err[key] = hist
        hist.append(err_sum / max(int(n), 1))

    def _lp_from_history(self, h: np.ndarray) -> float:
        """The error drop between the halves of `h`, or 0.0 unless it is
        SIGNIFICANT (relative gate) and MATERIAL (absolute floor)."""
        # _lp_gate_stage (telemetry only): 0 = no material drop, 1 = passed
        # the absolute floor, 2 = also passed the significance test.
        self._lp_gate_stage = 0
        half = len(h) // 2
        a, b = h[:half], h[half:]
        older = float(a.mean())
        recent = float(b.mean())
        drop = older - recent
        if not drop > 0.0:
            return 0.0
        # ABSOLUTE FLOOR: the drop must be a real fraction of the error
        # still outstanding — what makes a mastered bucket pay exactly 0
        # instead of an amplified jitter income.
        if not drop > (float(getattr(self, "lp_abs_frac", 0.0))
                       * max(recent, 1e-12)):
            return 0.0
        self._lp_gate_stage = 1
        z = float(getattr(self, "lp_sig_z", 0.0))
        if z <= 0.0:
            # LEGACY relative gate (revert / regression witness only): valid
            # for independent samples, passes 22-39% of a frozen model's
            # autocorrelated history. See __init__.
            sig = drop > float(getattr(self, "lp_sig_k", 0.5)) * float(h.std())
            if sig:
                self._lp_gate_stage = 2
            return drop if sig else 0.0
        n = len(h)
        dof = n - 2
        ss = float(((a - older) ** 2).sum() + ((b - recent) ** 2).sum())
        if dof <= 0:
            return 0.0
        var = ss / dof
        if var <= 0.0:
            self._lp_gate_stage = 2
            return drop            # noiseless and strictly lower: real
        d = np.concatenate([np.diff(a), np.diff(b)])
        r = 1.0 - float((d * d).mean()) / (2.0 * var) if len(d) else 0.0
        # Small-sample bias: an AR(1) estimate from m points reads LOW by
        # ~(1+3r)/m (Kendall). Uncorrected, 15-point halves under-read
        # rho 0.9 and A/B oscillation passed 3-5% instead of <=2%.
        r = r + (1.0 + 3.0 * max(r, 0.0)) / max(len(a), 1)
        r = min(max(r, 0.0), float(getattr(self, "lp_rho_max", 0.9)))
        infl = (1.0 + r) / (1.0 - r)
        se = (var * (1.0 / len(a) + 1.0 / len(b)) * infl) ** 0.5
        dof_eff = max(1.0, dof / infl)
        # Student-t quantile from the normal one (Cornish-Fisher, first
        # term): small or highly correlated histories need a larger t.
        t = z * (1.0 + (z * z + 1.0) / (4.0 * dof_eff))
        if drop > t * se:
            self._lp_gate_stage = 2
        return drop if drop > t * se else 0.0

    def _reset_gate_stats(self) -> None:
        self._gs_evaluated = 0
        self._gs_passed_abs = 0
        self._gs_passed_sig = 0
        self._gs_paid = 0
        self._gs_paid_sum = 0.0
        self._gs_collapsed = 0

    def pop_gate_stats(self) -> dict:
        """What the LP gates did on live calls since the last pop, then reset.

        evaluated        bucket ENTRIES scored (history >= lp_min_samples)
        passed_abs       of those, positive drop clearing the absolute floor
        passed_sig       of those, also clearing the significance test
        paid             entries whose raw LP > 0 (== passed_sig)
        collapsed_visits continuation steps folded into an open visit
        buckets          bucket histories currently held (a level, not a count)
        mean_paid        mean RAW (pre-normalisation) LP over paid entries,
                         None when nothing was paid
        Invariant: paid <= passed_sig <= passed_abs <= evaluated."""
        if not hasattr(self, "_gs_evaluated"):
            self._reset_gate_stats()
        out = {
            "evaluated": int(self._gs_evaluated),
            "passed_sig": int(self._gs_passed_sig),
            "passed_abs": int(self._gs_passed_abs),
            "paid": int(self._gs_paid),
            "collapsed_visits": int(self._gs_collapsed),
            "buckets": int(len(self._bucket_err)),
            "mean_paid": (float(self._gs_paid_sum / self._gs_paid)
                          if self._gs_paid else None),
        }
        self._reset_gate_stats()
        return out

    def compute_intrinsic_reward(self, obs, action, next_obs,
                                 update_state: bool = True,
                                 extra_error=None, stream_ids=None):
        """Learning progress over the agent's total prediction error.

        ---- THE FLOW RESIDUAL CHANNEL (2026-09-18) ----------------------
        `extra_error` is a per-sample error from the world model's flow head:
        how much of the change between the last two frames the
        action-conditioned warp could NOT explain. It is ADDED to the
        forward-model error that gets BUCKETED, so learning progress is
        measured over "how well do I predict this scene AND how it sweeps
        past me" rather than the latent prediction alone.

        WHY THIS IS NOT A NEW INCOME STREAM. Nothing here pays for error; LP
        pays for error GOING DOWN, and only when the drop clears both
        significance gates below. Adding a channel changes WHAT the agent can
        make progress on, not how much it is paid for standing anywhere.

        WHY IT CANNOT BE FARMED BY STARING. The flow residual requires
        ego-motion to be nonzero: no movement -> no predicted flow -> nothing
        to be wrong about -> zero contribution. Flat sky has no parallax and
        no texture for the warp to fail on. This is the property
        tests/_perspective_smoke.py asserts directly, and it is structural
        rather than a guard someone has to remember to re-open.

        `last_pred_error` IS DELIBERATELY NOT CONTAMINATED. It is documented
        as raw forward-model surprise and is read by skill wm_fidelity and the
        occlusion split; mixing a second channel into it would silently change
        what those two measure.
        """
        with torch.no_grad():
            features = self.encoder(obs)
            next_features = self.encoder(next_obs)
            predicted_next = self.forward_model(features, action)
            pred_error = F.mse_loss(
                predicted_next, next_features, reduction="none"
            ).mean(dim=-1)

            # RAW forward-model surprise, exposed for consumers that need
            # "how well does the WM predict THIS" rather than the shaped
            # curiosity signal (skill wm_fidelity; the occlusion split).
            # THE BASE CLASS SETS THIS TOO — but this subclass OVERRIDES
            # compute_intrinsic_reward entirely, so the parent's assignment
            # never ran and `last_pred_error` sat at its 0.0 init forever.
            # Consequences, both silent: the curiosity split read
            # `+0.0000 vs +0.0000 (ratio nan)`, and skill wm_fidelity — whose
            # whole purpose is measuring predictability — was fed a constant
            # zero. An overridden method silently orphans every side effect
            # the parent performed.
            self.last_pred_error = float(pred_error.mean().item())
            # counterfactual: what would we have predicted had we done nothing?
            _gate = None
            if self.action_conditional:
                _null = torch.zeros_like(action)
                if _null.dim() == 2 and _null.shape[1] > self.null_action:
                    _null[:, self.null_action] = 1.0      # discrete one-hot
                # continuous: all-zeros IS the null action, nothing to set
                _err_null = F.mse_loss(
                    self.forward_model(features, _null), next_features,
                    reduction="none").mean(dim=-1)
                _gate = torch.clamp(
                    (_err_null - pred_error) / (_err_null + 1e-8),
                    min=0.0, max=1.0)
                self.last_action_attribution = float(_gate.mean().item())
            # THE BUCKETED ERROR, which may carry the flow channel. Kept
            # separate from `pred_error` so `last_pred_error` above stays the
            # pure forward-model number its consumers expect.
            lp_error = pred_error
            if extra_error is not None:
                _xe = torch.as_tensor(
                    extra_error, dtype=lp_error.dtype, device=lp_error.device
                ).reshape(-1)
                if _xe.shape[0] == lp_error.shape[0]:
                    lp_error = lp_error + float(self.flow_error_weight) * _xe
                    self.last_flow_error = float(_xe.mean().item())
            errs = lp_error.detach().cpu().numpy()
            obs_np = obs.detach().cpu().numpy()
            streams = list(range(len(errs))) if stream_ids is None else list(stream_ids)
            if len(streams) != len(errs) or len(set(streams)) != len(streams):
                raise ValueError("stream_ids must be unique and match batch rows")
            lp = np.zeros(len(errs), dtype=np.float32)
            for i in range(len(errs)):
                key = self._bucket_key(obs_np[i], update_state=update_state)
                # VISIT COLLAPSE (see __init__): is this transition the ENTRY
                # into `key` for this row, or a continuation of the open
                # visit? Read-only calls (imagination) have no visits: every
                # row is scored against the closed history, as before, and
                # nothing below them is mutated.
                entry = True
                if self.lp_visit_collapse and update_state:
                    run = self._lp_runs.get(streams[i])
                    if run is not None and run[0] == key:
                        entry = False
                        self._gs_collapsed += 1
                        run[1] += float(errs[i])
                        run[2] += 1
                    else:
                        if run is not None:
                            self._close_visit(run)
                        self._lp_runs[streams[i]] = [key, float(errs[i]), 1]
                hist = self._bucket_err.get(key)
                if hist is None:
                    if not update_state:
                        continue        # read-only: unseen bucket -> LP 0
                    hist = deque(maxlen=self.lp_history)
                    self._bucket_err[key] = hist
                if entry and len(hist) >= self.lp_min_samples:
                    lp[i] = self._lp_from_history(
                        np.fromiter(hist, dtype=np.float64))
                    if update_state:
                        _st = self._lp_gate_stage
                        self._gs_evaluated += 1
                        self._gs_passed_abs += int(_st >= 1)
                        self._gs_passed_sig += int(_st >= 2)
                        if lp[i] > 0.0:
                            self._gs_paid += 1
                            self._gs_paid_sum += float(lp[i])
                if update_state and not self.lp_visit_collapse:
                    hist.append(float(errs[i]))         # counts next time
            if update_state:
                self._prune_buckets()

            reward = torch.clamp(
                torch.as_tensor(lp, dtype=torch.float32, device=obs.device),
                max=self.reward_clip)

            # Same adaptive normalization as the novelty arm, so the two arms feed
            # the reward mixer on a comparable scale (only the SHAPE differs):
            # median-centered, std-scaled, clamped at ZERO from below. LP is
            # mostly zeros, so the running median is ~0 and this reduces to a
            # std-scale — zero-progress transitions (including FIRST VISITS to
            # genuinely new states) get 0, never negative (the mean-centering
            # bug, CODE_AUDIT_2026-07.md §H.3), and there is no all-positive
            # baseline to farm (the 2026-07-12 regression).
            if update_state:
                self._update_stats(reward)
            if self._running_std > 1e-8:
                # std FLOOR (audit fix): the pool is mostly zeros (scout
                # singleton buckets + immature buckets), so an unfloored std
                # sits at noise-tick magnitude and amplifies residual jitter
                # to O(clip). The floor caps that gain; real drops still
                # saturate the clip.
                reward = (reward - self._running_median) / max(
                    self._running_std, 1e-3)
                reward = torch.clamp(reward, 0.0, self.reward_clip)
            # ATTRIBUTION LAST, so it scales the finished curiosity signal
            # rather than distorting the LP statistics that feed the running
            # median/std (those must keep tracking the TRUE forward error, or
            # the normalizer would chase its own gate).
            if _gate is not None:
                reward = reward * _gate
            # Scale AFTER normalization (pre-normalization scaling cancels).
            return reward * self.reward_scale
