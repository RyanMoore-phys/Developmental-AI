"""Smoke: imagination curiosity — the drive that does not die.

Contracts (the ones that matter for safety and for the user's goal):
  1. BOREDOM GATE: while external curiosity is healthy (global_lp high) the
     imagination bonus is exactly 0 — real novelty always outranks daydreaming.
  2. NEVER-ZERO: once external curiosity is dead (global_lp ~ 0) the bonus is
     STRICTLY POSITIVE, and it stays positive even when the model imagines NO
     reward at all (the disagreement term is what never dies).
  3. DECAY: the daydream (imagined-reward) term shrinks with experience, while
     the disagreement term does NOT — so the floor is durable but the
     "imagining nice things" part cannot become the point of living.
  4. NO-WIREHEAD SHAPE: a model that imagines a HUGE reward but is CERTAIN
     (zero disagreement) earns strictly less than the cap, and the imagined
     term is bounded — imagining a jackpot cannot dominate.
  5. SAFETY: bounded by max_bonus, never negative, never raises.

Run: PYTHONPATH=. python tests/_imagination_curiosity_smoke.py
"""
import sys

import torch

sys.path.insert(0, ".")

from developmental_ai.curiosity.imagination_curiosity import (
    ImaginationCuriosity)


class FakeWM(torch.nn.Module):
    """World model stub: controllable disagreement and imagined reward."""

    def __init__(self, spread: float, reward: float):
        super().__init__()
        self.spread, self.reward = spread, reward

    def imagine_trajectory(self, initial_state, policy_fn, horizon,
                           knowledge=None):
        K = initial_state["h"].shape[0]
        D = initial_state["h"].shape[1]
        # latents disagree by `spread` across the K rollouts
        base = torch.zeros(K, horizon, D)
        noise = torch.randn(K, horizon, D) * self.spread
        return {"latents": base + noise,
                "rewards": torch.full((K, horizon, 1),
                                      self.reward / max(1, horizon)),
                "continues": torch.ones(K, horizon, 1)}


def state(K=1, D=16):
    return {"h": torch.zeros(K, D), "z": torch.zeros(K, D)}


def policy_fn(latent):
    return torch.zeros(latent.shape[0], 4)


def make(**kw):
    base = dict(enabled=True, every=1, rollouts=8, horizon=10,
                disagreement_weight=0.30, imagined_reward_weight=0.10,
                imagined_reward_halflife=300_000, lp_reference=0.05,
                max_bonus=0.5)
    base.update(kw)
    return ImaginationCuriosity(**base)


def main() -> None:
    torch.manual_seed(0)
    wm_rich = FakeWM(spread=1.0, reward=5.0)     # uncertain + rewarding
    wm_certain = FakeWM(spread=0.0, reward=50.0)  # CERTAIN but "jackpot"

    # ---- 1. BOREDOM GATE: healthy external curiosity -> exactly 0 ----
    ic = make()
    b_busy = ic.bonus(wm_rich, state(), policy_fn, timestep=0,
                      global_lp=0.20)            # >> lp_reference
    assert b_busy == 0.0, f"daydreamed while the world was teaching: {b_busy}"

    # ---- 2. NEVER-ZERO once external curiosity is dead ----
    ic = make()
    b_bored = ic.bonus(wm_rich, state(), policy_fn, timestep=0,
                       global_lp=0.0)
    assert b_bored > 0.0, f"no drive left when bored: {b_bored}"

    # ...and STILL positive with ZERO imagined reward (disagreement carries it)
    ic = make()
    b_nore = ic.bonus(FakeWM(spread=1.0, reward=0.0), state(), policy_fn,
                      timestep=0, global_lp=0.0)
    assert b_nore > 0.0, (
        f"drive died when the model imagined no reward ({b_nore}) — the "
        f"disagreement term is supposed to be the never-dying part")

    # ---- 3. DECAY: daydream term shrinks; disagreement does NOT ----
    ic = make()
    early = ic.bonus(wm_rich, state(), policy_fn, 0, 0.0)
    ic2 = make()
    late = ic2.bonus(wm_rich, state(), policy_fn, 3_000_000, 0.0)
    assert late < early, f"daydream bonus did not decay: {early} -> {late}"
    # the surviving floor is the disagreement term, still > 0 after decay
    assert late > 0.0, f"everything decayed to zero: {late}"

    # ---- 4. NO-WIREHEAD SHAPE: certain-but-jackpot is bounded and small ----
    ic = make()
    b_jack = ic.bonus(wm_certain, state(), policy_fn, 0, 0.0)
    assert b_jack < ic.max_bonus, f"jackpot hit the cap: {b_jack}"
    # imagining a 50-reward jackpot with NO uncertainty must not beat
    # genuinely-uncertain exploration with a modest reward
    ic3 = make()
    b_unc = ic3.bonus(wm_rich, state(), policy_fn, 0, 0.0)
    assert b_unc > b_jack, (
        f"certain jackpot ({b_jack:.4f}) out-paid uncertain exploration "
        f"({b_unc:.4f}) — that is the wireheading gradient")

    # ---- 5. SAFETY: bounded, non-negative, never raises ----
    for lp in (0.0, 0.01, 0.05, 1.0):
        v = make().bonus(wm_rich, state(), policy_fn, 0, lp)
        assert 0.0 <= v <= 0.5, f"out of bounds at lp={lp}: {v}"

    class Broken:
        training = False

        def eval(self):
            pass

        def train(self, mode=True):
            pass

        def imagine_trajectory(self, *a, **k):
            raise RuntimeError("boom")

    assert make().bonus(Broken(), state(), policy_fn, 0, 0.0) == 0.0, \
        "a failing world model must not break the run"

    # disabled -> exactly 0
    assert make(enabled=False).bonus(wm_rich, state(), policy_fn, 0, 0.0) == 0.0

    # ---- 6. WIRING: the loop's imagination policy must feed the actor
    #      [obs, gated_knowledge], exactly like prospection. Feeding raw obs
    #      is a silent shape bomb that only fires in a live run (seen:
    #      "mat1 and mat2 shapes cannot be multiplied (8x49152 and
    #      49250x256)" — 49250-49152 = 98 = the goal-broadcast width).
    loop_src = open("developmental_ai/core/developmental_loop.py").read()
    imag = loop_src[loop_src.index("IMAGINATION CURIOSITY"):]
    imag = imag[:imag.index("EXTERNAL CURIOSITY")]
    assert "_augment(" in imag, (
        "the imagination policy does not augment the actor input with the "
        "goal broadcast — this is the 49152-vs-49250 shape bomb")
    assert "_prep_knowledge(" in imag, \
        "imagination policy never prepares a knowledge vector"

    # ---- 7. SAFETY: imagination must never touch the replay reward LABEL ----
    assert "intrinsic[0] = intrinsic[0] + _ib" in loop_src, \
        "imagination bonus is not going into the intrinsic channel"
    assert "prim_extrinsic = rewards[0] + _ib" not in loop_src, (
        "imagination bonus reached prim_extrinsic — that is the replay reward "
        "LABEL, which trains the WM reward head: the wireheading loop")

    # ================= PAST-STATE BASELINE (2026-09-25) ==================
    # THE LIVE INCIDENT. Both SkyBots dug into a cave and stopped acting:
    #   infra/stuck: LEVEL 3 — seg_extrinsic=0, cells_delta=0, events=0
    #   infra/episodic: break:dirt @(-28,-3) 196144 steps ago
    #   infra/ledger: total=+8.51 | imagination=69%, novelty=13%
    # 196k steps without breaking a block, and 69% of ALL income from this
    # module. Cause: `dis` was normalised against a DECAYING running maximum
    #   self._dis_scale = max(self._dis_scale * 0.999, dis_raw)
    # so as the model learned the cave and dis_raw fell, the scale fell with
    # it and `dis` climbed back to ~1.0. The module docstring promises the
    # bonus "self-extinguishes LOCALLY"; it could not. These four contracts
    # are that promise, made testable.
    # ---- A. a state you have fully learned stops paying -----------------
    ic = make(baseline_every=1, baseline_samples=2, past_states=8)
    wm_flat = FakeWM(spread=0.5, reward=0.0)
    seq = [ic.bonus(wm_flat, state(), policy_fn, 0, 0.0) for _ in range(40)]
    assert seq[-1] < seq[0] * 0.25, (
        f"A. constant disagreement did not extinguish: {seq[0]:.4f} -> "
        f"{seq[-1]:.4f}. This is the cave: same place, same surprise, "
        f"still paying.")
    print(f"  A. learned state extinguishes: {seq[0]:.4f} -> {seq[-1]:.4f}")

    # ---- B. ...but genuine novelty still pays (never dies GLOBALLY) -----
    # STATE-DEPENDENT, and that is the whole point. A FakeWM with one global
    # `spread` cannot test this: it makes the REMEMBERED states uncertain
    # too, so the baseline rises with it and the excess is ~0. That version
    # of this contract passed at 0.0026 and proved nothing. The question is
    # "is HERE stranger than THERE", so the fake has to answer differently
    # per state.
    class PlaceWM(FakeWM):
        """spread is read from the state itself: h==0 is home, h==1 is new."""

        def imagine_trajectory(self, initial_state, policy_fn, horizon,
                               knowledge=None):
            here = float(initial_state["h"][0, 0].item())
            self.spread = 0.5 + 4.5 * here      # 0.5 at home, 5.0 away
            return super().imagine_trajectory(
                initial_state, policy_fn, horizon, knowledge)

    def place(v, D=16):
        h = torch.zeros(1, D); h[0, 0] = v
        return {"h": h, "z": torch.zeros(1, D)}

    ic_b = make(baseline_every=1, baseline_samples=2, past_states=8)
    wm_b = PlaceWM(spread=0.5, reward=0.0)
    home = [ic_b.bonus(wm_b, place(0.0), policy_fn, 0, 0.0)
            for _ in range(40)]
    away = ic_b.bonus(wm_b, place(1.0), policy_fn, 0, 0.0)
    assert home[-1] < home[0] * 0.25, (
        f"B. home did not extinguish: {home[0]:.4f} -> {home[-1]:.4f}")
    assert away > max(0.05, home[-1] * 10), (
        f"B. a state 10x more uncertain THAN THE REMEMBERED ONES paid "
        f"{away:.4f} against an exhausted home of {home[-1]:.4f} — local "
        f"extinction became global death")
    print(f"  B. novel place still pays: {away:.4f} vs exhausted home "
          f"{home[-1]:.4f}")

    # ---- C. COLD START keeps the original behaviour ---------------------
    # An agent with no remembered past must still be curious. Paying 0 here
    # would be a latch of exactly the kind __init__ warns about.
    ic_cold = make(baseline_every=1, baseline_samples=4, past_states=8)
    b_cold = ic_cold.bonus(FakeWM(spread=0.5, reward=0.0), state(),
                           policy_fn, 0, 0.0)
    assert b_cold > 0.0, (
        f"C. cold start paid {b_cold} — an agent with no memory was "
        f"silenced by a baseline it has not built yet")
    assert ic_cold._dis_baseline is None, \
        "C. baseline was set from a buffer too small to be a baseline"
    print(f"  C. cold start unchanged: {b_cold:.4f}, baseline not yet set")

    # ---- D. cost is bounded -------------------------------------------
    # The baseline is the only cost this adds. Paying it EVERY probe would
    # multiply imagination cost by baseline_samples on a GPU that has only
    # just stopped running out of memory.
    class CountingWM(FakeWM):
        calls = 0

        def imagine_trajectory(self, *a, **k):
            CountingWM.calls += 1
            return super().imagine_trajectory(*a, **k)

    # baseline_every=2 so the baseline ACTUALLY ENGAGES inside the loop.
    # At 10 it needs 4 stored states x 10 probes each = 40 probes before it
    # can run at all, so a 30-probe test measured the un-baselined path and
    # proved nothing — the first version of this contract did exactly that.
    ic_c = make(baseline_every=2, baseline_samples=3, past_states=16)
    wm_c = CountingWM(spread=0.5, reward=0.0)
    N = 40
    for _ in range(N):
        ic_c.bonus(wm_c, state(), policy_fn, 0, 0.0)
    assert ic_c._dis_baseline is not None, \
        "D. baseline never engaged — this contract would be vacuous"
    # N "here" rollouts + at most one 3-sample baseline every 2 probes
    ceiling = N + (N // 2) * 3
    assert CountingWM.calls <= ceiling, (
        f"D. {CountingWM.calls} imagine calls in {N} probes (ceiling "
        f"{ceiling}) — the baseline is being recomputed per probe")
    print(f"  D. cost bounded: {CountingWM.calls} rollout calls / {N} "
          f"probes (ceiling {ceiling}), baseline live")

    print(f"[imagination-curiosity-smoke] ALL PASS: boredom-gated "
          f"(busy={b_busy}), never-zero (bored={b_bored:.4f}, "
          f"no-imagined-reward={b_nore:.4f}), decays "
          f"({early:.4f}->{late:.4f}), uncertain>{'jackpot'} "
          f"({b_unc:.4f}>{b_jack:.4f}), bounded+safe")


if __name__ == "__main__":
    main()
