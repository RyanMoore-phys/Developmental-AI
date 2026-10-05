"""Paired progress: probe churn, partial failure and drift must not pay.

2026-10 review: the old aggregate EMA compared different exams and paid
remaining error reductions repeatedly. These tests pin matched identities,
fixed observable targets, model versions, and shadow-only event delivery.

2026-10-05 second review (contracts E-G):
  E. RESTORE IS EVIDENCE, NOT CLOCK. load_state_dict used __dict__.update,
     restoring _last_eval_step from the old run while total_timesteps
     restarts at 0: `step - _last_eval_step` stayed negative, so no probe
     was re-evaluated until the new run overtook the old count (a latch,
     CLAUDE.md 4.1) — and config (_eval_every, _capacity, significance_z)
     was silently overwritten by the file. Pinned: after a restore at step
     0 the next evaluate RUNS and pairs against the saved baselines; config
     comes from the constructor; a smaller capacity drops surplus probes
     and their baselines.
  F. A REJECTED STATE TOUCHES NOTHING. Inconsistent tables raise before
     any field is assigned.
  G. SIDECAR FILES CANNOT CRASH-LOOP BOOT. A truncated / wrong-schema file
     -> 'cold' for that component, exactly one warning, the file renamed to
     .corrupt-<ts> (kept, never deleted). Absent (older checkpoints) ->
     'absent', silent. Writes are atomic and leave no temp files.
"""
from dataclasses import replace
import glob
import logging
import os
import pickle
import tempfile
import torch
from developmental_ai.infra.progress_curiosity import (
    ProbeSetProgress, EvidenceCredit, atomic_pickle_dump, load_pickle_sidecar)


class _Count(logging.Handler):
    def __init__(self):
        logging.Handler.__init__(self)
        self.records = []

    def emit(self, record):
        self.records.append(record)


def _logger():
    log = logging.getLogger('progress_sidecar_smoke')
    log.propagate = False
    for h in list(log.handlers):
        log.removeHandler(h)
    h = _Count()
    log.addHandler(h)
    log.setLevel(logging.INFO)
    return log, h


def filled():
    p = ProbeSetProgress(capacity=4, eval_every=1)
    for i in range(4):
        p.maybe_add_probe({'target': torch.tensor([[float(i)]])}, evidence_ref=f'ev:{i}')
    return p


def main():
    p = filled()
    assert p.evaluate(lambda b: 2+b['target'].item(), 0, model_version='v0') == 0
    # Easier replacement cannot create improvement in the surviving probes.
    for i in range(30):
        p.maybe_add_probe({'target': torch.tensor([[-1.]])})
    assert p.evaluate(lambda b: 2+b['target'].item(), 1, model_version='v1') == 0
    print('  A. reservoir churn does not create progress')
    p = filled()
    p.evaluate(lambda b: 2+b['target'].item(), 0, model_version='v0')
    def partial(b):
        if b['target'].item() == 3:
            raise ValueError('unavailable hard probe')
        return 2+b['target'].item()
    assert p.evaluate(partial, 1, model_version='v1') == 0
    assert p.eval_errors == 1
    print('  B. losing a hard probe does not create progress')
    p = filled()
    p.evaluate(lambda b: 1., 0, model_version='v0')
    assert p.evaluate(lambda b: .25, 1, model_version='v1') > 0
    e = p.last_event
    assert e.improvement == .75 and e.probe_ids and e.previous_versions == ('v0',)
    assert p.rate() == 0
    assert p.evaluate(lambda b: 0., 2, model_version='v1') is None
    assert p.evaluate(lambda b: .25, 3, model_version='v2') == 0
    assert p.evaluate(lambda b: 1., 4, model_version='v3') == 0
    assert p.last_event.forgetting == .75
    assert p.evaluate(lambda b: .25, 5, model_version='v4') == 0
    # A representation/objective change starts a new exam, never a windfall.
    assert p.evaluate(lambda b: .01, 6, model_version='v5', objective='new-objective') == 0
    print('  C. same-scene improvement measured; duplicate versions/relearning/target changes pay zero')
    q = filled(); q.load_state_dict(pickle.loads(pickle.dumps(p.state_dict())))
    # The version cursor is NOT restored (contract E): only evidence is.
    assert q._last_version is None and q._last_eval_step is None and q.rate() == 0
    assert q._previous == p._previous and q._ids == p._ids
    assert q.evaluate(lambda b: .01, 7, model_version='v6', objective='new-objective') == 0
    ledger = EvidenceCredit(cap=.1)
    assert ledger.claim(e, 'ev:0', 1) == 0  # Shadow events cannot be paid.
    live = replace(e, shadow=False)
    assert ledger.claim(live, 'unrelated', 1) == 0
    assert ledger.claim(live, 'ev:0', 1) == .1
    assert ledger.claim(live, 'ev:0', 2) == 0
    restored = EvidenceCredit(); restored.load_state_dict(ledger.state_dict())
    assert restored.claim(live, 'ev:0', 2) == 0
    assert EvidenceCredit(max_age=1).claim(live, 'ev:0', 10) == 0
    print('  D. resume, finite evidence credit, expiry and no ambient income')
    try:
        p.maybe_add_probe({'target': torch.zeros(1)}, split='heldout')
        raise AssertionError('held-out leakage')
    except ValueError:
        pass

    # ---- E. restore after a step reset re-evaluates; config wins ---------
    old = ProbeSetProgress(capacity=4, eval_every=512)
    for i in range(4):
        old.maybe_add_probe({'target': torch.tensor([[float(i)]])}, evidence_ref=f'ev:{i}')
    old.evaluate(lambda b: 1., 100000, model_version='adam_step=900')
    assert old._last_eval_step == 100000
    state = pickle.loads(pickle.dumps(old.state_dict()))
    new = ProbeSetProgress(capacity=4, eval_every=512)
    new.load_state_dict(state)
    assert new.due(0), 'restored clock latched the cadence'
    got = new.evaluate(lambda b: .5, 0, model_version='adam_step=901')
    assert got is not None and got > 0, got           # ran AND paired with saved baselines
    assert new.last_event.previous_versions == ('adam_step=900',)
    assert new.rate() == 0
    # Falsifier: the pre-fix __dict__ restore really did latch here.
    latched = ProbeSetProgress(capacity=4, eval_every=512)
    latched.__dict__.update(copy_state(state))
    assert latched.evaluate(lambda b: .5, 0, model_version='adam_step=901') is None
    cfg = ProbeSetProgress(capacity=3, eval_every=7, significance_z=1.0)
    cfg.load_state_dict(state)
    assert (cfg._eval_every, cfg._capacity, cfg.significance_z) == (7, 3, 1.0)
    assert len(cfg._probes) == len(cfg._ids) == len(cfg._refs) == 3
    assert set(cfg._previous) == set(cfg._ids) and set(cfg._best) == set(cfg._ids)
    print('  E. restore at step 0 re-evaluates against saved baselines; '
          'config (eval_every/capacity/z) comes from config, never the file')

    # ---- F. a rejected state mutates nothing --------------------------------
    bad = pickle.loads(pickle.dumps(state))
    bad['state']['_ids'] = bad['state']['_ids'][:-1]
    fresh = ProbeSetProgress(capacity=4, eval_every=512)
    try:
        fresh.load_state_dict(bad)
        raise AssertionError('inconsistent probe tables accepted')
    except ValueError:
        pass
    assert fresh._probes == [] and fresh._previous == {} and fresh._objective is None
    print('  F. inconsistent state rejected before any field is assigned')

    # ---- G. sidecar files: atomic write; corrupt -> cold + warning + kept ---
    d = tempfile.mkdtemp(prefix='progress_sidecar_')
    path = os.path.join(d, 'progress_probes.pkl')
    log, h = _logger()
    fresh = ProbeSetProgress(capacity=4, eval_every=512)
    assert load_pickle_sidecar(path, fresh.load_state_dict, 'probes', log=log) == 'absent'
    assert not h.records, 'an absent (older-checkpoint) sidecar must be silent'
    atomic_pickle_dump(path, old.state_dict())
    assert sorted(os.listdir(d)) == ['progress_probes.pkl'], os.listdir(d)  # no temp left
    ok = ProbeSetProgress(capacity=4, eval_every=512)
    assert load_pickle_sidecar(path, ok.load_state_dict, 'probes', log=log) == 'loaded'
    assert ok._ids == old._ids and not h.records
    with open(path, 'rb') as f:
        data = f.read()
    with open(path, 'wb') as f:
        f.write(data[:len(data)//2])                   # a torn non-atomic write
    fresh = ProbeSetProgress(capacity=4, eval_every=512)
    assert load_pickle_sidecar(path, fresh.load_state_dict, 'probes', log=log) == 'cold'
    warns = [r for r in h.records if r.levelno == logging.WARNING]
    assert len(warns) == 1 and 'COLD START' in warns[0].getMessage(), h.records
    assert not os.path.exists(path)
    kept = glob.glob(path + '.corrupt-*')
    assert len(kept) == 1 and open(kept[0], 'rb').read() == data[:len(data)//2]
    assert fresh._probes == [] and fresh.due(0) is False  # cold == as constructed
    # a VALID pickle the loader rejects (schema) is the same cold start
    atomic_pickle_dump(path, {'schema': 99})
    h.records.clear()
    assert load_pickle_sidecar(path, fresh.load_state_dict, 'probes', log=log) == 'cold'
    assert len(h.records) == 1 and len(glob.glob(path + '.corrupt-*')) == 2
    # the component keeps working after a cold start (no latch)
    for i in range(3):
        fresh.maybe_add_probe({'target': torch.tensor([[float(i)]])})
    assert fresh.evaluate(lambda b: 1., 0, model_version='v0') == 0.0
    print('  G. sidecar: atomic write, absent silent, torn/wrong-schema file '
          '-> cold start + 1 warning + file kept aside')
    print('[infra_progress_curiosity_smoke] ALL PASS')


def copy_state(state):
    import copy
    return copy.deepcopy(state['state'])


if __name__ == '__main__':
    main()
