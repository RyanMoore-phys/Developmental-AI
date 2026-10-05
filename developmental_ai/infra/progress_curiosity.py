"""Paired development-probe progress, independent of action rewards.

Only identical probe identities with the same observable objective can be
compared. New/replaced probes establish baselines. Failed probes cannot make
an aggregate easier. Versions must advance; repeated readings are not learning.
New signals are shadow-only. Explicit evidence credit is a separate API and
is not connected to the live reward mixer.
"""
from collections import deque
from dataclasses import asdict, dataclass
import copy
import logging
import math
import os
import pickle
import random
import tempfile
import time
import uuid
from typing import Any, Callable, Dict
import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---- CHECKPOINT SIDECARS (2026-10-05 review) -------------------------------
# curiosity_visits.pkl and progress_probes.pkl sit beside the torch
# checkpoints. Two ways they could kill a run, both fixed here:
#  * a NON-ATOMIC write interrupted by a crash/OOM leaves a truncated file;
#  * an UNGUARDED load of that file (or of a valid file whose prototype size
#    no longer matches the config) raised inside load_checkpoint, so boot
#    died, the supervisor relaunched, boot died again — forever. A guard
#    with no reachable escape is a latch (CLAUDE.md 4.1).
# Escape path: an unreadable sidecar costs ONLY its own component a cold
# start, with one warning, and the file is renamed aside (never deleted) so
# a human can inspect it; the next checkpoint writes a fresh one.
def atomic_pickle_dump(path, obj):
    """tmp file in the same directory + fsync + os.replace: readers see the
    old file or the new one, never a torn one."""
    path = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + '.',
                               suffix='.tmp', dir=directory)
    try:
        with os.fdopen(fd, 'wb') as f:
            pickle.dump(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    try:  # make the rename itself durable (best effort; not on every FS)
        dfd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    except OSError:
        pass


def load_pickle_sidecar(path, apply: Callable[[Any], None], label,
                        log=None):
    """Load `path` and hand it to `apply`. Returns 'absent', 'loaded' or
    'cold'. 'absent' (older checkpoints never had the file) is silent.
    'cold': unreadable or rejected by `apply` -> one warning, the file is
    moved to `<path>.corrupt-<timestamp>` and the component keeps its fresh
    state. `apply` MUST validate before it mutates, so a rejection leaves
    the component exactly as constructed."""
    log = log or logger
    path = os.fspath(path)
    if not os.path.exists(path):
        return 'absent'
    try:
        with open(path, 'rb') as f:
            state = pickle.load(f)
        apply(state)
        return 'loaded'
    except Exception as exc:
        aside = '%s.corrupt-%s' % (path, time.strftime('%Y%m%dT%H%M%S'))
        n = 1
        while os.path.exists(aside):
            aside = '%s.corrupt-%s-%d' % (path, time.strftime('%Y%m%dT%H%M%S'), n)
            n += 1
        try:
            os.replace(path, aside)
        except OSError as mv_exc:
            aside = '%s (could not move aside: %s)' % (path, mv_exc)
        log.warning('%s sidecar unusable (%s: %s) -> COLD START for this '
                    'component only; file kept at %s', label,
                    type(exc).__name__, exc, aside)
        return 'cold'


@dataclass(frozen=True)
class ProgressEvent:
    event_id: str
    previous_versions: tuple
    model_version: str
    objective: str
    probe_ids: tuple
    evidence_refs: tuple
    improvement: float
    forgetting: float
    standard_error: float
    step: int
    shadow: bool = True


class EvidenceCredit:
    """Bounded single-use credit for explicitly attributed experience.

    Not wired into PPO. No credit for shadow events, expired events, unrelated
    evidence, or repeated claims. The consumed set never evicts: on capacity
    exhaustion this ledger closes until an explicit new run/ledger is started.
    """
    def __init__(self, cap=0.1, max_age=512, capacity=10000):
        if not math.isfinite(cap) or cap < 0 or max_age < 0 or capacity < 1:
            raise ValueError('invalid evidence credit budget')
        self.cap, self.max_age, self.capacity = cap, max_age, capacity
        self.consumed = set()

    def claim(self, event, evidence_ref, step):
        if (event.shadow or event.event_id in self.consumed
                or len(self.consumed) >= self.capacity
                or not 0 <= step - event.step <= self.max_age
                or evidence_ref not in event.evidence_refs):
            return 0.0
        self.consumed.add(event.event_id)
        return min(self.cap, max(0.0, event.improvement))

    def state_dict(self):
        return {'consumed': sorted(self.consumed)}

    def load_state_dict(self, state):
        self.consumed = set(state['consumed'])


class ProbeSetProgress:
    def __init__(self, capacity=12, eval_every=512, ema_beta=0.3,
                 norm_decay=0.999, shadow=True, significance_z=2.33):
        if capacity < 3 or eval_every < 1 or not math.isfinite(significance_z) or significance_z < 0:
            raise ValueError('invalid paired-probe configuration')
        self._capacity, self._eval_every = int(capacity), int(eval_every)
        # Kept as accepted arguments for old configs; EMA comparisons are gone.
        self.shadow = bool(shadow)
        self.significance_z = float(significance_z)
        self._rng = random.Random(0x5EED)
        self._probes, self._ids, self._refs = [], [], []
        self._offers = 0
        self._previous, self._best = {}, {}
        self._objective = None
        self._last_eval_step = self._last_success_step = None
        self._last_version = None
        self._step = 0
        self._rate = 0.0
        self._running_max = 0.0
        self._norm_decay = norm_decay
        self.last_mean_loss = None
        self.last_progress = 0.0
        self.last_forgetting = 0.0
        self.last_error = None
        self.eval_errors = 0
        self.last_event = None
        self.events = deque(maxlen=128)
        self._run_id = uuid.uuid4().hex
        self._serial = 0

    def maybe_add_probe(self, batch: Dict[str, Any], *, evidence_ref=None, split='dev'):
        if split != 'dev':
            raise ValueError('progress probes must be development evidence')
        try:
            probe = {k: v[0:1].detach().cpu().clone()
                     for k, v in batch.items() if torch.is_tensor(v)}
            if not probe:
                return False
            self._offers += 1
            j = len(self._probes) if len(self._probes) < self._capacity else self._rng.randrange(self._offers)
            if j >= self._capacity:
                return False
            pid = f'{self._run_id}:probe:{self._offers}'
            if j == len(self._probes):
                self._probes.append(probe); self._ids.append(pid); self._refs.append(evidence_ref)
            else:
                old = self._ids[j]
                self._previous.pop(old, None); self._best.pop(old, None)
                self._probes[j], self._ids[j], self._refs[j] = probe, pid, evidence_ref
            return True
        except Exception as exc:
            self.last_error = repr(exc)
            return False

    def evaluate(self, loss_fn, step, device=None, *, model_version=None,
                 objective='observable-mse-v1'):
        step = int(step)
        version = str(step if model_version is None else model_version)
        if not self.due(step) or version == self._last_version:
            return None
        self._last_eval_step = step
        if objective != self._objective:
            self._previous.clear(); self._best.clear()
            self._objective = objective
        losses, paired = [], []
        with torch.no_grad():
            for pid, ref, probe in zip(self._ids, self._refs, self._probes):
                try:
                    batch = probe if device is None else {k: v.to(device) for k, v in probe.items()}
                    val = float(loss_fn(batch))
                    if not math.isfinite(val) or val < 0:
                        raise ValueError('probe loss must be finite and nonnegative')
                    losses.append(val)
                    if pid in self._previous:
                        old_version, old_loss = self._previous[pid]
                        paired.append((pid, ref, old_version, old_loss-val,
                                       self._best[pid]-val))
                    self._previous[pid] = (version, val)
                    self._best[pid] = min(self._best.get(pid, val), val)
                except Exception as exc:
                    self.eval_errors += 1; self.last_error = repr(exc)
        self.last_event = None
        self._rate = self.last_progress = 0.0
        if not losses:
            return None
        self.last_mean_loss = sum(losses)/len(losses)
        self._last_success_step, self._last_version = step, version
        if len(paired) < 3:
            return 0.0
        drops = np.array([x[4] for x in paired])  # Improvement over best-ever score.
        se = float(drops.std(ddof=1)/math.sqrt(len(drops)))
        raw = max(0.0, float(drops.mean()) - self.significance_z*se)
        self.last_forgetting = max(0.0, -float(np.mean([x[3] for x in paired])))
        self._running_max = max(self._running_max*self._norm_decay, raw)
        self.last_progress = raw/self._running_max if self._running_max else 0.0
        self._serial += 1
        event = ProgressEvent(f'{self._run_id}:event:{self._serial}',
            tuple(sorted(set(x[2] for x in paired))), version, objective,
            tuple(x[0] for x in paired), tuple(x[1] for x in paired if x[1] is not None),
            raw, self.last_forgetting, se, step, self.shadow)
        self.last_event = event; self.events.append(event)
        return self.last_progress

    def due(self, step):
        """Cadence check, cheap enough for every step: callers test this
        BEFORE taking the WM lock or computing a model version."""
        step = int(step)
        if len(self._probes) < 3:
            return False
        return (self._last_eval_step is None
                or step - self._last_eval_step >= self._eval_every)

    def tick(self, step):
        self._step = int(step)

    def rate(self):
        # Measurement is never an ambient wage. Use EvidenceCredit explicitly.
        return 0.0

    def state_dict(self):
        return {'schema': 2, 'state': copy.deepcopy(self.__dict__)}

    # EVIDENCE ONLY (2026-10-05 review). The old load did
    # __dict__.update(saved), which also restored the CLOCK: _last_eval_step
    # from the previous run while total_timesteps restarts at 0, so
    # `step - _last_eval_step` stayed negative and no probe was re-evaluated
    # until the new run overtook the old count — a latch (CLAUDE.md 4.1). It
    # also overwrote config (_eval_every, _capacity, significance_z). Now:
    # evidence comes from the file, config from the constructor, and the
    # clock/version cursor start fresh, so the first due evaluation of the
    # new process re-scores every probe against its saved baseline.
    _EVIDENCE = ('_probes', '_ids', '_refs', '_offers', '_previous', '_best',
                 '_objective', '_running_max', 'eval_errors',
                 'last_mean_loss', 'last_forgetting', '_run_id', '_serial')

    def load_state_dict(self, state):
        if not isinstance(state, dict) or state.get('schema') != 2:
            raise ValueError('unsupported paired-progress state')
        saved = state.get('state')
        if not isinstance(saved, dict):
            raise ValueError('paired-progress state has no body')
        missing = [k for k in self._EVIDENCE if k not in saved]
        if missing:
            raise ValueError('paired-progress state lacks %s' % missing)
        ev = copy.deepcopy({k: saved[k] for k in self._EVIDENCE})
        probes, ids, refs = list(ev['_probes']), list(ev['_ids']), list(ev['_refs'])
        if not len(probes) == len(ids) == len(refs) or len(set(ids)) != len(ids):
            raise ValueError('paired-progress probe tables are inconsistent')
        if not all(isinstance(p, dict) and p and all(torch.is_tensor(v) for v in p.values())
                   for p in probes):
            raise ValueError('paired-progress probes must be tensor dicts')
        running_max = float(ev['_running_max'])
        if not math.isfinite(running_max) or running_max < 0:
            raise ValueError('paired-progress normaliser is invalid')
        offers, eval_errors = int(ev['_offers']), int(ev['eval_errors'])
        forgetting, serial = float(ev['last_forgetting']), int(ev['_serial'])
        run_id = str(ev['_run_id'])
        saved_rng = saved.get('_rng')
        rng_state = saved_rng.getstate() if isinstance(saved_rng, random.Random) else None
        events = list(saved.get('events', ()))
        previous, best = dict(ev['_previous']), dict(ev['_best'])
        # Config wins: a smaller capacity drops the surplus probes (and their
        # baselines), never the reverse.
        for pid in ids[self._capacity:]:
            previous.pop(pid, None); best.pop(pid, None)
        probes, ids, refs = probes[:self._capacity], ids[:self._capacity], refs[:self._capacity]
        # --- validated; apply ---
        self._probes, self._ids, self._refs = probes, ids, refs
        self._offers = max(offers, len(probes))
        self._previous, self._best = previous, best
        self._objective = ev['_objective']
        self._running_max = running_max
        self.eval_errors = eval_errors
        self.last_mean_loss = ev['last_mean_loss']
        self.last_forgetting = forgetting
        self._run_id, self._serial = run_id, serial
        if rng_state is not None:
            self._rng.setstate(rng_state)
        self.events = deque(events, maxlen=self.events.maxlen)
        # Fresh clock and cursor (see block comment).
        self._last_eval_step = self._last_success_step = None
        self._last_version = None
        self._step = 0
        self.last_event = None
        self.last_error = None
        self._rate = self.last_progress = 0.0

    @property
    def stats(self):
        return {'probes': len(self._probes), 'last_mean_loss': self.last_mean_loss,
                'last_progress': self.last_progress, 'forgetting': self.last_forgetting,
                'eval_errors': self.eval_errors, 'rate': 0.0, 'shadow': self.shadow,
                'event': None if self.last_event is None else asdict(self.last_event)}
