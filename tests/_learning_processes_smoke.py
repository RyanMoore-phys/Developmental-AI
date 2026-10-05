"""Regression contracts for 2026-10 learning-process hardening.

Exercises actual LP stream lifecycle, observable forecasting, validation
selection and shared costs. No live environment or training host required.

2026-10-05 second review (contracts F-G):
  F. LP VISIT SIDECAR CANNOT CRASH-LOOP BOOT. A prototype-size change
     (lp_pixel_pool / image_channels) made the saved prototypes
     incomparable with live frames; the unguarded load either raised in
     load_checkpoint (supervisor relaunch, forever — CLAUDE.md 4.1) or
     crashed the first _bucket_key. Pinned: the mismatch is rejected BEFORE
     any field is touched; through load_pickle_sidecar it is a cold start
     for that component with one warning and the file kept aside; a
     matching module loads and keeps bucketing; an absent sidecar (older
     checkpoint) is silent and leaves the module as constructed.
  G. LOOP WIRING, by source (this tier has no gymnasium, so the loop
     cannot be imported — same method as _foundation_shadow_smoke B):
     both live bodies call _curiosity_reward with IDENTICAL text (count 2,
     §4.2) and never the raw compute_intrinsic_reward; no
     record_reward("progress") path exists; a restart closes the pre-crash
     visit (no `valid=not bad`); the progress cadence is checked before the
     model-version scan; sidecars are written atomically and loaded through
     the guarded helper; evaluation uses its own stream id namespace;
     gui_dwell_weight is validated at __init__.
"""
import copy
import glob
import logging
import os
import pickle
import re
import tempfile
import numpy as np
import torch
from developmental_ai.curiosity.learning_progress import LearningProgressCuriosity
from developmental_ai.core.reward_components import gui_costs
from developmental_ai.foundation.runtime.observable import predict_observation, sequence_loss
from developmental_ai.world_model.rssm import WorldModel
from developmental_ai.infra.progress_curiosity import (
    atomic_pickle_dump, load_pickle_sidecar)

LOOP = os.path.join('developmental_ai', 'core', 'developmental_loop.py')
EVAL = os.path.join('developmental_ai', 'core', 'evaluation.py')
LIVE_CALL = ('intrinsic = self._curiosity_reward(\n'
             '                obs_t, action_tensor, next_obs_t, extra_error=_fr,\n'
             '                ended=dones, invalid=[bool(restarted[i]) or bool(\n'
             '                    (step_infos[i] or {}).get("env_restarted")) '
             'for i in range(n)])')


def _body(src, name):
    i = src.index('    def %s(' % name)
    j = src.index('\n    def ', i + 10)
    return src[i:j]


def pixel_lp(pool):
    return LearningProgressCuriosity(
        obs_dim=3 * 64 * 64, action_dim=3, feature_dim=8, hidden_dim=8,
        pixel_obs=True, image_channels=3, image_size=64, lp_pixel_pool=pool,
        lp_history=8, lp_min_samples=4)


def lp():
    return LearningProgressCuriosity(obs_dim=4, action_dim=3, feature_dim=4,
                                     hidden_dim=8, lp_history=8)


def main():
    torch.set_num_threads(1)
    for kw in ({'lp_rho_max': 1}, {'lp_rho_max': float('nan')}, {'lp_sig_z': -1}):
        try:
            LearningProgressCuriosity(obs_dim=4, action_dim=3, **kw)
            raise AssertionError('invalid setting accepted')
        except ValueError:
            pass
    p = lp()
    o = torch.zeros(2, 4); o[1] = 10
    a = torch.eye(3)[:2]
    p.compute_intrinsic_reward(o, a, o+1, stream_ids=['a', 'b'])
    p.compute_intrinsic_reward(o.flip(0), a.flip(0), (o+1).flip(0), stream_ids=['b', 'a'])
    assert p._lp_runs['a'][2] == p._lp_runs['b'][2] == 2
    p.end_visits(['a'])
    assert sum(map(len, p._bucket_err.values())) == 1 and 'b' in p._lp_runs
    p.end_visits(['b'], valid=False)
    assert sum(map(len, p._bucket_err.values())) == 1 and not p._lp_runs
    q = lp(); q.load_visit_state_dict(pickle.loads(pickle.dumps(p.visit_state_dict())))
    assert dict(q._bucket_err) == dict(p._bucket_err) and not q._lp_runs
    assert q._running_std == p._running_std
    print('  A. settings, explicit stream identity, valid/reset boundaries, resume')

    w = WorldModel(obs_dim=4, action_dim=2, stochastic_size=2,
                   stochastic_classes=3, deterministic_size=8, hidden_dim=16)
    state = w.rssm.initial_state(1, torch.device('cpu'))
    action = torch.tensor([[1., 0.]])
    rng = torch.random.get_rng_state().clone()
    weights = copy.deepcopy(w.state_dict())
    pred = predict_observation(w, state, action)
    assert pred.shape == (1, 4) and torch.equal(rng, torch.random.get_rng_state())
    assert all(torch.equal(v, w.state_dict()[k]) for k, v in weights.items())
    batch = {'observations': torch.randn(1, 4, 4),
             'actions': torch.tensor([[[1., 0.]]*4]), 'continues': torch.ones(1,4,1)}
    rng = torch.random.get_rng_state().clone()
    l1, l2 = sequence_loss(w, batch), sequence_loss(w, batch)
    assert l1 == l2 and np.isfinite(l1) and torch.equal(rng, torch.random.get_rng_state())
    batch['continues'].zero_()
    try:
        sequence_loss(w, batch)
        raise AssertionError('reset splice accepted')
    except ValueError:
        pass
    print('  B. observable fixed targets, no RNG/weight changes, invalid pairs rejected')

    from developmental_ai.foundation.mechanisms._nets import train
    m = torch.nn.Linear(1, 1)
    opt = torch.optim.Adam(m.parameters(), lr=.01)
    seen, vals = [], iter([3., 1., 2., 4.])
    def validate():
        seen.append(copy.deepcopy(m.state_dict()))
        return next(vals)
    hist = train(list(m.parameters()), lambda i: m(torch.ones(len(i),1)).square().mean(),
                 4, epochs=10, batch_size=4, lr=.01, seed=0,
                 optimizer=opt, validation_fn=validate, module=m, patience=2)
    assert len(hist) == 4
    assert all(torch.equal(v, seen[1][k]) for k,v in m.state_dict().items())
    assert int(next(iter(opt.state.values()))['step']) == 2
    print('  C. early stop restores selected weights AND optimizer moments')

    run, phi, total = 0, 0., 0.
    for flag in [True]*10+[False]:
        c = gui_costs(flag, run, phi, weight=.05, dwell_steps=3, step_cost=.002, grace_steps=2)
        run, phi = c.run, c.potential
        total += c.total
    assert abs(total + .016) < 1e-12 and run == 0
    assert gui_costs(False, 10, None, weight=.05).total == 0
    print('  D. shared GUI accounting preserves cycle/refund and dwell charge')

    from tools.shadow_report import validation_split, observable_scores
    rows = [{'env':'env', 'stream':'s', 'episode':str(e), 'seq':t} for e in range(5) for t in range(8)]
    fit, val, basis = validation_split(rows)
    assert set(r['episode'] for r in fit).isdisjoint(r['episode'] for r in val)
    scores = observable_scores({'rows':[{'part':'heldout', 'stream':'s', 'pay':{
        'observable_score':{'status':'scored','target':'pov','mse':9.,'persistence_mse':1.}}}]})
    assert scores['groups'][0]['mse'] == 9.
    print('  E. episode validation isolation; confident wrong forecast is scored as wrong')

    # ---- F. LP visit sidecar: size validation, cold start, absence -------
    src4 = pixel_lp(4)
    g = torch.Generator().manual_seed(0)
    frames = torch.rand(2, 3 * 64 * 64, generator=g)
    acts = torch.eye(3)[:2]
    for _ in range(3):
        src4.compute_intrinsic_reward(frames, acts, frames, stream_ids=[0, 1])
        frames = torch.rand(2, 3 * 64 * 64, generator=g)
    src4.end_visits()
    assert src4._protos and all(v.shape == (48,) for v in src4._protos.values())
    state = src4.visit_state_dict()
    other = pixel_lp(2)                                 # 3*2*2 = 12-d prototypes
    try:
        other.load_visit_state_dict(pickle.loads(pickle.dumps(state)))
        raise AssertionError('mismatched prototype size accepted')
    except ValueError as e:
        assert 'expects (12,)' in str(e), e
    assert other._protos == {} and other._bucket_err == {} and other._proto_next_id == 0
    d = tempfile.mkdtemp(prefix='lp_sidecar_')
    path = os.path.join(d, 'curiosity_visits.pkl')
    log = logging.getLogger('lp_sidecar_smoke'); log.propagate = False
    seen = []
    h = logging.Handler(); h.emit = seen.append; log.addHandler(h)
    assert load_pickle_sidecar(path, other.load_visit_state_dict, 'visits', log=log) == 'absent'
    assert not seen and other._protos == {}             # older checkpoint: silent
    atomic_pickle_dump(path, state)
    assert load_pickle_sidecar(path, other.load_visit_state_dict, 'visits', log=log) == 'cold'
    assert len(seen) == 1 and seen[0].levelno == logging.WARNING
    assert not os.path.exists(path) and len(glob.glob(path + '.corrupt-*')) == 1
    assert other._protos == {}
    other.compute_intrinsic_reward(frames, acts, frames, stream_ids=[0, 1])  # still runs
    same = pixel_lp(4)
    atomic_pickle_dump(path, state)
    assert load_pickle_sidecar(path, same.load_visit_state_dict, 'visits', log=log) == 'loaded'
    assert set(same._protos) == set(src4._protos) and len(seen) == 1
    same.compute_intrinsic_reward(frames, acts, frames, stream_ids=[0, 1])  # bucketing works
    print('  F. LP sidecar: prototype size validated before mutation; mismatch '
          '-> cold start + 1 warning + kept aside; absent silent; match loads')

    # ---- G. loop wiring by source ------------------------------------------
    src = open(LOOP).read()
    assert src.count(LIVE_CALL) == 2, src.count(LIVE_CALL)
    for name in ('_run_episode_parallel', '_collect_segment'):
        b = _body(src, name)
        assert b.count('self._curiosity_reward(') == 1 and LIVE_CALL in b, name
        assert 'self.curiosity.compute_intrinsic_reward(' not in b, name
    assert not re.search(r'record_reward\(\s*["\']progress["\']', src)
    cr = _body(src, '_curiosity_reward')
    assert 'valid=not bad' not in cr and 'self._curiosity_boundary([i])' in cr
    assert src.index('self._progress.due(self.total_timesteps)') < src.index(
        'model_version=self._world_model_version()')
    assert src.count('model_version=self._world_model_version()') == 1
    sv, ld = _body(src, '_save_checkpoint_locked'), _body(src, 'load_checkpoint')
    assert sv.count('atomic_pickle_dump(') == 2 and 'pickle.dump(' not in sv
    assert ld.count('load_pickle_sidecar(') == 2 and 'pickle.load(' not in ld
    assert '_gui_reward_components' not in src
    assert 'curiosity.gui_dwell_weight must be finite and >= 0' in _body(src, '__init__')
    ev = open(EVAL).read()
    assert 'stream_ids=[EVAL_STREAM_ID]' in ev and 'end_visits([EVAL_STREAM_ID], valid=False)' in ev
    assert 'compute_intrinsic_reward(\n                    obs_t, act_t, next_obs_t\n' not in ev
    print('  G. wiring: identical _curiosity_reward x2, no progress payment, '
          'restart closes pre-crash visit, cadence before version, guarded '
          'sidecars, eval stream namespace, gui weight validated at init')
    print('[learning_processes_smoke] ALL PASS')


if __name__ == '__main__':
    main()
