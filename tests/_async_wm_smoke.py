"""Smoke: async world-model trainer mechanics (no env, no real WM).

Binds the REAL DevelopmentalAI._ensure_wm_trainer to a bare stub (no
__init__) with _train_world_model stubbed, and drives the exact ticket-site
semantics used in _collect_segment. Contracts:
  1. a ticket runs training on the trainer thread; metrics published by
     ref-swap; busy flag set during / cleared after
  2. skip-if-busy: cadence points during a running consolidation post
     NOTHING (no backlog) — total executions stay 1
  3. a crashing _train_world_model is logged and the trainer SURVIVES for
     the next ticket
  4. stop event + join shuts the thread down
Run: PYTHONPATH=. python tests/_async_wm_smoke.py
"""
import queue
import sys
import threading
import time

sys.path.insert(0, ".")

from developmental_ai.core.developmental_loop import DevelopmentalAI


def make_stub(train_fn):
    stub = object.__new__(DevelopmentalAI)   # no __init__: mechanics only
    stub._use_async_wm = True
    stub._wm_trainer_thread = None
    stub._wm_ticket = queue.Queue(maxsize=1)
    stub._wm_trainer_stop = threading.Event()
    stub._wm_busy = threading.Event()
    stub._wm_param_lock = threading.RLock()
    stub._wm_metrics_box = None
    stub._train_world_model = train_fn
    return stub


def post_ticket(stub) -> bool:
    """The exact ticket-site semantics from _collect_segment."""
    if not stub._wm_busy.is_set():
        try:
            stub._wm_ticket.put_nowait(1)
            return True
        except Exception:
            return False
    return False


def wait_until(cond, timeout=5.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.01)
    return False


def main() -> None:
    # 1. ticket -> trained -> metrics ref-swap; busy toggles around it
    calls = []

    def slow_train():
        calls.append(1)
        time.sleep(0.35)
        return {"loss": float(len(calls))}

    s = make_stub(slow_train)
    DevelopmentalAI._ensure_wm_trainer(s)
    assert post_ticket(s)
    assert wait_until(lambda: s._wm_busy.is_set(), 2.0), "busy never set"

    # 2. skip-if-busy: cadence points during the 0.35s train post nothing
    skipped = [post_ticket(s) for _ in range(5)]
    assert not any(skipped), skipped
    assert wait_until(lambda: s._wm_metrics_box is not None, 3.0)
    assert wait_until(lambda: not s._wm_busy.is_set(), 2.0)
    assert calls == [1], calls
    assert s._wm_metrics_box == {"loss": 1.0}

    # next cadence point after completion trains again on fresher data
    assert post_ticket(s)
    assert wait_until(lambda: len(calls) == 2 and not s._wm_busy.is_set(), 3.0)
    assert s._wm_metrics_box == {"loss": 2.0}

    # 3. crash survival: a raising train must not kill the trainer
    crashes = []

    def crashing_train():
        crashes.append(1)
        if len(crashes) == 1:
            raise RuntimeError("boom")
        return {"ok": 1.0}

    s2 = make_stub(crashing_train)
    DevelopmentalAI._ensure_wm_trainer(s2)
    assert post_ticket(s2)
    assert wait_until(lambda: len(crashes) == 1 and not s2._wm_busy.is_set(),
                      3.0)
    assert s2._wm_metrics_box is None          # crash published nothing
    assert s2._wm_trainer_thread.is_alive(), "trainer died on exception"
    assert post_ticket(s2)
    assert wait_until(lambda: s2._wm_metrics_box == {"ok": 1.0}, 3.0)

    # 4. shutdown: stop + join
    for st in (s, s2):
        st._wm_trainer_stop.set()
        st._wm_trainer_thread.join(timeout=5)
        assert not st._wm_trainer_thread.is_alive(), "trainer failed to stop"

    print("[async-wm-smoke] ALL PASS: ticket/skip-if-busy/ref-swap, crash "
          "survival, clean shutdown")


if __name__ == "__main__":
    main()
