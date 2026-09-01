"""Gate / GateRegistry smoke — the anti-latch contract.

Contracts:
  1. close/open idempotency: re-closing keeps the ORIGINAL since-step and
     first reason; open clears; a later re-close starts a fresh clock.
  2. overdue fires only PAST max_closed_steps and the alarm line names the
     gate, age, budget, reason, and the declared reopen condition.
  3. set_state edge detection: open->closed records step+reason once,
     repeated same-state calls are no-ops, closed->open clears.
  4. alarms()/table() formats are stable: "name: OPEN" and
     "name: CLOSED since=<s> (<n> steps, max=<m>) reason=<r>".
  5. a gate created via set_state, driven overdue, then reopened stops
     alarming — the alarm channel must clear, not latch (that would be
     latch #10, inside the latch detector).
"""

from developmental_ai.infra.gate import Gate, GateRegistry


def test_close_open_idempotency():
    g = Gate("wm_training", reopen="loss ema < 0.5", max_closed_steps=100)
    assert not g.is_closed
    g.close(5, "loss spike")
    g.close(40, "second call must not win")          # idempotent re-close
    assert g.is_closed
    # Original clock: at step 5+100=105 exactly at budget -> silent;
    # if the re-close at 40 had reset the clock this would also be silent
    # at 141, so check 141 fires AND carries the FIRST reason.
    assert g.overdue(105) is None
    msg = g.overdue(141)
    assert msg is not None and "loss spike" in msg, msg
    assert "second call" not in msg, msg
    g.open(150)
    g.open(151)                                      # idempotent re-open
    assert not g.is_closed and g.overdue(10**9) is None
    # A fresh closure starts a fresh clock with a fresh reason.
    g.close(200, "new episode of trouble")
    assert g.overdue(250) is None
    msg2 = g.overdue(301)
    assert msg2 is not None and "new episode of trouble" in msg2, msg2


def test_overdue_budget_and_message():
    g = Gate("exploration", reopen="novelty ema recovers above floor",
             max_closed_steps=50)
    assert g.overdue(999) is None                    # open: never overdue
    g.close(100, "budget exhausted")
    assert g.overdue(100) is None                    # 0 steps closed
    assert g.overdue(150) is None                    # == max: within budget
    msg = g.overdue(151)                             # max+1: fires
    assert msg is not None
    assert "\n" not in msg                           # one-line contract
    for needle in ("exploration", "51", "50", "budget exhausted",
                   "novelty ema recovers above floor"):
        assert needle in msg, (needle, msg)


def test_set_state_edge_detection():
    reg = GateRegistry()
    reg.set_state("lp_floor", True, step=10, reason="lp below floor",
                  reopen="lp ema rises", max_closed_steps=5)
    g = reg.set_state("lp_floor", True, step=30,
                      reason="later tick must not win")  # same-state no-op
    assert g.is_closed
    msg = g.overdue(30)                              # 20 steps > max 5
    assert msg is not None and "since step 10" in msg, msg
    assert "lp below floor" in msg and "later tick" not in msg, msg
    reg.set_state("lp_floor", False, step=40)        # closed->open edge
    assert not g.is_closed and g.overdue(10**6) is None
    reg.set_state("lp_floor", False, step=41)        # same-state no-op
    assert not g.is_closed
    # create-or-get: bare-name access returns the SAME gate and keeps the
    # real declaration (defaults must not clobber it).
    same = reg.gate("lp_floor")
    assert same is g and same.reopen == "lp ema rises"
    assert same.max_closed_steps == 5


def test_alarms_and_table_formats():
    reg = GateRegistry()
    reg.gate("a", reopen="signal returns", max_closed_steps=100)
    reg.gate("b", reopen="operator resets", max_closed_steps=4)
    reg.gate("b").close(3, "r")
    lines = reg.table(12)
    assert lines == ["a: OPEN",
                     "b: CLOSED since=3 (9 steps, max=4) reason=r"], lines
    alarms = reg.alarms(12)
    assert len(alarms) == 1, alarms
    assert "'b'" in alarms[0] and "operator resets" in alarms[0], alarms
    assert reg.alarms(5) == []                       # within budget: quiet


def test_reopened_gate_stops_alarming():
    reg = GateRegistry()
    reg.set_state("guard", True, step=0, reason="cold start",
                  reopen="warmup complete", max_closed_steps=10)
    assert len(reg.alarms(11)) == 1                  # driven overdue
    reg.set_state("guard", False, step=12)           # the re-arming path
    assert reg.alarms(10**7) == []                   # alarm CLEARS
    assert reg.gate("guard").overdue(10**7) is None
    assert reg.table(12) == ["guard: OPEN"]


if __name__ == "__main__":
    tests = [
        (test_close_open_idempotency,
         "close/open idempotent; original since-step + first reason kept"),
        (test_overdue_budget_and_message,
         "overdue fires only past max; alarm names reopen condition"),
        (test_set_state_edge_detection,
         "set_state edge-detects; same-state no-ops; get keeps declaration"),
        (test_alarms_and_table_formats,
         "alarms/table stable formats"),
        (test_reopened_gate_stops_alarming,
         "set_state-created gate reopened stops alarming"),
    ]
    for i, (fn, contract) in enumerate(tests, 1):
        print(f"[infra-gate] {i}. {contract}")
        fn()
    print("[infra-gate] ALL PASS")
