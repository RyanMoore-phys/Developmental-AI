"""Smoke: tiny-batch PPO collapse guard + offered-vs-chosen observability.

THE BUG (measured live 2026-07-25). The PPO update TRIGGER counts env steps
(`rollout_env_steps` = sum of taus) so option use cannot delay updates. Correct
— but under heavy option use ONE stored row spans tau~40 env steps, so a
1024-env-step update fires on as few as ~25 rows. `train_step` has NO
minibatching (`policy.batch_size` is never read there), so it then ran 10 FULL
passes over those ~25 advantage-normalized samples. The option logits were
crushed and the meta-policy stopped picking options ENTIRELY (~38,940 masked
draws, zero picks), which SELF-LOCKS: a policy that never invokes an option
never generates option experience to learn from. Options fired 219 times and
then never again for 32,800 steps.

The fix for one starvation problem created another, which is exactly why the
second contract below exists: the two failure modes ("never offered" vs
"offered, never chosen") are INDISTINGUISHABLE from the old logs — both just
show `total=` frozen — and they need opposite fixes.

Run: PYTHONPATH=. python tests/_option_collapse_smoke.py
"""
import sys

sys.path.insert(0, ".")

import numpy as np

AC = "developmental_ai/policy/actor_critic.py"
OPT = "developmental_ai/policy/options.py"
LOOP = "developmental_ai/core/developmental_loop.py"


def epochs_for(rows: int, requested: int = 10, rows_per_epoch: int = 32) -> int:
    """Mirror of the guard in train_step (kept in sync by the assert below)."""
    return max(1, min(requested, rows // rows_per_epoch))


def main() -> None:
    ac = open(AC).read()
    opt = open(OPT).read()
    loop = open(LOOP).read()

    # ---- A. THE COLLAPSE GUARD EXISTS AND IS IN train_step ---------------
    assert "_ROWS_PER_EPOCH = 32" in ac, "tiny-batch guard missing"
    assert "n_epochs = max(1, min(_req, _rows // _ROWS_PER_EPOCH))" in ac, \
        "epoch count no longer scales with sample count"
    # it must sit INSIDE train_step, before the epoch loop
    ts = ac.index("def train_step")
    assert ts < ac.index("_ROWS_PER_EPOCH") < ac.index("for _ in range(n_epochs)"), \
        "guard is not between train_step's start and its epoch loop"

    # ---- B. THE ARITHMETIC THE BUG TURNED ON ----------------------------
    # the measured pathology: 1024 env steps at tau~40 => ~25 rows
    rows_at_saturation = 1024 // 40
    assert rows_at_saturation == 25
    assert epochs_for(rows_at_saturation) == 1, (
        f"{rows_at_saturation} rows must collapse to ONE epoch; 10 full-batch "
        f"passes over 25 samples is what crushed the option logits")
    # a healthy no-option rollout is unaffected (parity with old behaviour)
    assert epochs_for(1024) == 10, "large batches must keep all 10 epochs"
    assert epochs_for(320) == 10, "320 rows is exactly 10*32 — still 10 epochs"
    assert epochs_for(319) == 9, "just below the boundary loses one epoch"
    # never zero, never negative, monotonic in rows
    prev = 0
    for r in (0, 1, 2, 31, 32, 64, 100, 512, 4096):
        e = epochs_for(r)
        assert 1 <= e <= 10, f"{r} rows -> {e} epochs is out of range"
        assert e >= prev or r < 32, "epoch count must be monotonic in rows"
        prev = e

    # ---- C. OFFERED vs CHOSEN IS OBSERVABLE ------------------------------
    # Without this, "gate never offered" and "policy never picked" look
    # IDENTICAL in the logs (total= just stops rising). Opposite fixes.
    for counter in ("decisions_open", "decisions_with_offer", "option_picks"):
        assert counter in opt, f"{counter} not tracked"
    assert "self.decisions_open += 1" in opt
    assert "if bool(mask[P:].any()):" in opt, \
        "offered-ness is not measured from the ACTUAL mask handed to the policy"
    assert "if int(a) >= P:" in opt and "self.option_picks += 1" in opt, \
        "option picks are not counted"
    assert '"offered_vs_chosen"' in opt, "counters never reach the snapshot"
    assert "Option offers:" in loop, "never printed — unobservable in a log"

    # PRODUCER AND CONSUMER MUST ACTUALLY CONNECT. The first cut of this fix
    # published the counters in snapshot() but READ them via a non-existent
    # stats(); the broad except swallowed the AttributeError and the line
    # silently never printed. Checking producer and consumer SEPARATELY (as
    # this test originally did) passes right through that. Bind them by name.
    import inspect

    from developmental_ai.policy.options import OptionExecutor
    _method = None
    for _name in ("snapshot", "stats"):
        if hasattr(OptionExecutor, _name):
            _src = inspect.getsource(getattr(OptionExecutor, _name))
            if "offered_vs_chosen" in _src:
                _method = _name
                break
    assert _method, "no executor method publishes offered_vs_chosen"
    assert f"self.option_executor.{_method}()" in loop, (
        f"counters are published by {_method}() but the log site does not "
        f"call it — the telemetry cannot reach the log")

    # ...and a swallowed telemetry error must never look like a zero reading.
    # Scope this to the telemetry block itself — the file legitimately has
    # other `except: pass` guards, and asserting over the whole file both
    # fails for the wrong reason and would forbid unrelated valid code.
    # Anchor the region STRUCTURALLY (enclosing try: -> next section marker),
    # not by a fixed character window: a +/-1200 char window silently fell out
    # of range the moment the block grew, failing for the wrong reason.
    _marker = loop.index("Option offers:")
    _blk = loop[loop.rindex("try:", 0, _marker):
                loop.index("CHOP-COMPLETION MEASUREMENT", _marker)]
    assert "pass   # telemetry" not in _blk, \
        "telemetry error is swallowed silently — a vanished line is " \
        "indistinguishable from a legitimately zero metric"
    assert "UNAVAILABLE" in _blk, \
        "a broken telemetry line must SAY it is broken, not vanish"
    assert "OFFERED BUT NEVER CHOSEN" in loop, \
        "the collapse signature is not called out where a reader would see it"

    # counting order must be: count the decision, then the pick — a pick can
    # only happen at a counted decision, so picked <= with_offer <= decisions
    i_dec = opt.index("self.decisions_open += 1")
    i_off = opt.index("if bool(mask[P:].any()):")
    i_sel = opt.index("a, info = policy.select_action(")
    i_pick = opt.index("if int(a) >= P:")
    assert i_dec < i_off < i_sel < i_pick, \
        "counters are not ordered decision -> offer -> select -> pick"

    # ---- D. THE INVARIANT THE COUNTERS MUST SATISFY ----------------------
    # simulate: an option can only be picked when one was offered
    rng = np.random.default_rng(0)
    dec = off = pick = 0
    P = 10
    for _ in range(5000):
        mask = np.zeros(P + 5, dtype=bool)
        mask[:P] = True
        if rng.random() < 0.6:                    # a slot is offered
            mask[P + rng.integers(0, 5)] = True
        dec += 1
        if mask[P:].any():
            off += 1
        a = int(rng.choice(np.flatnonzero(mask)))
        if a >= P:
            pick += 1
    assert pick <= off <= dec, f"invariant violated: {pick} <= {off} <= {dec}"
    assert off > 0 and pick > 0, "sanity: the simulation should exercise both"

    print(f"[option-collapse-smoke] ALL PASS: {rows_at_saturation} rows at "
          f"option saturation -> {epochs_for(rows_at_saturation)} epoch (was "
          f"10 full-batch passes, which crushed the option logits), 1024 rows "
          f"-> 10 epochs unchanged; offered/chosen counters ordered correctly "
          f"and printed, so 'never offered' and 'never chosen' are now "
          f"distinguishable in the log")


if __name__ == "__main__":
    main()
