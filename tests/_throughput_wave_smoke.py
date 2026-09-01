"""Throughput + learning-cadence wave (2026-08-23).

WHAT THIS DEFENDS

A. THE PER-STEP COST IS OP COUNT, NOT ARITHMETIC.
   Profiled on the LIVE modules at the LIVE config: one env step issues
   ~2350 aten ops at batch num_envs=2 — `curiosity.train_step` 1519,
   `compute_intrinsic_reward` 295, `world_model.encoder`+`rssm.observe_step`
   275, `policy.select_action` 260 — plus ~25 CPU<->GPU sync points. The
   recorded GPU utilisation is 2%: the device is IDLE and we are paying
   kernel-launch and pipeline-drain latency. Nothing here is a big model
   (the whole curiosity module is 1.25M params with a CNN encoder).

   A1 batches the curiosity update: same gradient information, K-fold fewer
   launches (1519 -> 380 ops/step at K=4, -> 95 at K=16). It BATCHES and
   never SKIPS — forward-model error IS the curiosity reward, so training
   less often would slow novelty decay, a live reward-economy change.

   A3 fuses select_action's three `.item()` calls into one transfer. Each
   `.item()` blocks the CPU until the GPU drains, and the VLM shares that
   device, so a sync can wait behind a 1.5s inference.

B. A STARVED POLICY WAS REPORTED AS A HEALTHY ONE.
   PPO's tiny-batch collapse guard caps epochs at `rows // 32`. Under SMDP
   options one ROW spans tau env steps, so at tau~40 a 1024-env-step update
   holds ~25 rows -> 25//32 = 0 -> ONE full-batch gradient step, about one
   per 5 minutes of play. The guard is correct (10 epochs on ~25 rows once
   crushed the option logits into a self-locking policy), but it logged the
   reduction at DEBUG, and the loop's segment print annotated ANY reduction
   as "stopped early (good: the policy hit its movement budget)" — which
   describes target_kl early-stopping. target_kl is 0.0, withdrawn
   2026-08-05. So the only reachable cause was starvation, reported as
   health.

Run: PYTHONPATH=. python tests/_throughput_wave_smoke.py
"""
import os
import sys
import types

sys.path.insert(0, ".")

import numpy as np
import torch

LOOP = os.path.join("developmental_ai", "core", "developmental_loop.py")
AC = os.path.join("developmental_ai", "policy", "actor_critic.py")


# ---------------------------------------------------------------- A1 -----
def _stub_loop(k):
    """Minimal object exercising the real _curiosity_train unbound method."""
    from developmental_ai.core.developmental_loop import DevelopmentalAI
    calls = []

    class _Cur:
        def train_step(self, o, a, n):
            calls.append(int(o.shape[0]))
            return {"icm_total": float(o.shape[0])}

    s = types.SimpleNamespace(_curiosity_train_every=k, _cur_train_buf=[],
                              _last_icm_metrics={}, curiosity=_Cur())
    return DevelopmentalAI._curiosity_train, s, calls


def test_A1_curiosity_batching():
    fn, s, calls = _stub_loop(1)
    for _ in range(5):
        fn(s, torch.zeros(2, 4), torch.zeros(2, 3), torch.zeros(2, 4))
    assert calls == [2, 2, 2, 2, 2], calls
    print(f"  A1. train_every=1 -> one update per step, batch {calls[0]} "
          f"(byte-identical to the pre-change path)")

    fn, s, calls = _stub_loop(4)
    for _ in range(12):
        fn(s, torch.zeros(2, 4), torch.zeros(2, 3), torch.zeros(2, 4))
    assert calls == [8, 8, 8], calls
    assert s._cur_train_buf == [], "buffer not drained after firing"
    print(f"  A1b. train_every=4 over 12 steps -> {len(calls)} updates of "
          f"batch {calls[0]} — same 24 transitions, 4x fewer launches")

    # the steps that do NOT train must still return usable metrics
    fn, s, calls = _stub_loop(4)
    m0 = fn(s, torch.zeros(2, 4), torch.zeros(2, 3), torch.zeros(2, 4))
    assert m0 == {}, "first non-training step should yield the empty prior"
    for _ in range(3):
        m = fn(s, torch.zeros(2, 4), torch.zeros(2, 3), torch.zeros(2, 4))
    assert m.get("icm_total") == 8.0
    m_idle = fn(s, torch.zeros(2, 4), torch.zeros(2, 3), torch.zeros(2, 4))
    assert m_idle.get("icm_total") == 8.0, (
        "an idle step must carry the last metrics forward, not None — "
        "curiosity_loss is read unconditionally")
    print("  A1c. non-training steps carry the last metrics forward "
          "(curiosity_loss is read every step)")

    # no autograd graph may be retained across steps
    fn, s, _ = _stub_loop(4)
    leaf = torch.zeros(2, 4, requires_grad=True)
    fn(s, leaf * 2, torch.zeros(2, 3), torch.zeros(2, 4))
    assert s._cur_train_buf[0][0].requires_grad is False, (
        "buffered tensors keep their graph — this leaks memory and ties the "
        "batch to graphs the optimizer already consumed")
    print("  A1d. buffered transitions are detached (no retained graph)")


def test_A1_no_duplicated_body_drift():
    src = open(LOOP).read()
    assert src.count("icm_metrics = self._curiosity_train(") == 3, (
        "all three stepping bodies must route through the shared helper")
    # `self.curiosity.train_step(` must survive ONLY inside the helper (twice:
    # the pass-through and the batched call). Anywhere else means a stepping
    # body still trains directly — that is how the six previous
    # duplicated-body bugs happened. NB match on the assignment target, since
    # `_last_icm_metrics` ends with the substring `icm_metrics`.
    assert src.count("self.curiosity.train_step(") == 2, (
        f"expected 2 direct train_step calls (both inside _curiosity_train), "
        f"found {src.count('self.curiosity.train_step(')}")
    i_def = src.index("def _curiosity_train(")
    i_next = src.index("def _apply_explore_boost(")
    assert src[i_def:i_next].count("self.curiosity.train_step(") == 2, (
        "a direct train_step call lives OUTSIDE the shared helper")
    assert "def _curiosity_train(" in src
    print("  A1e. all 3 stepping bodies share ONE helper — this feature "
          "cannot drift between duplicated bodies by construction")

    cfg = open(os.path.join("configs", "minecraft_skybot.yaml")).read()
    assert "train_every:" in cfg, "config lost the curiosity.train_every knob"
    import yaml
    k = int((yaml.safe_load(cfg).get("curiosity", {}) or {}).get(
        "train_every", 1))
    assert k == 1, (
        f"train_every is {k}: K>1 is NOT identical (one Adam step on K*N "
        f"samples != K steps on N) and cannot be validated without a live "
        f"run. Ship at 1; raise it with a pod to watch.")
    print(f"  A1f. config ships train_every={k} (off) — the mechanism is "
          f"staged, the live behaviour is unchanged")


# ---------------------------------------------------------------- A3 -----
def test_A3_single_sync_in_select_action():
    from developmental_ai.policy.actor_critic import StandaloneActorCritic
    torch.manual_seed(0)
    p = StandaloneActorCritic(obs_dim=16, action_dim=5, hidden_dim=32,
                              continuous=False)
    obs = np.random.RandomState(0).rand(16).astype(np.float32)

    a, info = p.select_action(obs, deterministic=True)
    assert isinstance(a, int) and 0 <= a < 5, (a, type(a))
    assert isinstance(info["log_prob"], float)
    assert isinstance(info["value"], float)

    # EXACTNESS: recompute the same quantities the long way and require an
    # exact match — the fused transfer must not perturb any value.
    with torch.no_grad():
        t = torch.FloatTensor(obs).unsqueeze(0)
        aug = p._augment(p._encode(t), p._prep_knowledge(None),
                         p._prep_proprio(None))
        act_ref, lp_ref = p.actor.get_action(aug, deterministic=True)
        # `_value_of`, NOT `p.critic` (2026-09-01). The critic now predicts
        # symlog(return) and every READ decodes with symexp, so the raw head
        # output is no longer a value in reward space. This assertion is
        # about the fused device->host transfer not perturbing anything, so
        # the reference has to be the same quantity select_action returns —
        # comparing against the undecoded head would be testing the value
        # parameterization by accident, and would have "passed" only while
        # the two spaces happened to coincide.
        v_ref = p._value_of(aug)
    assert int(act_ref.reshape(-1)[0].item()) == a, (act_ref, a)
    assert float(lp_ref.reshape(-1)[0].item()) == info["log_prob"]
    assert float(v_ref.reshape(-1)[0].item()) == info["value"]
    print(f"  A3. select_action exact: action={a} "
          f"log_prob={info['log_prob']:.6f} value={info['value']:.6f} "
          f"— identical to the unfused computation")

    # the masked (options) branch must be exact too
    mask = np.array([True, False, True, False, True])
    a2, info2 = p.select_action(obs, deterministic=True, action_mask=mask)
    assert mask[a2], f"masked action {a2} is not a valid choice"
    print(f"  A3b. masked/meta path still respects the mask (chose {a2})")

    src = open(AC).read()
    i = src.index("def select_action")
    j = src.index("def compute_last_value")
    # count CODE, not prose: the explanatory comment names `.item()` several
    # times, and a test that cannot tell a comment from a call is a test that
    # will one day pass for the wrong reason
    body = "\n".join(ln for ln in src[i:j].splitlines()
                     if not ln.strip().startswith("#"))
    # The A2 self-check (verify_feats) legitimately reads one scalar to report
    # feature drift, but it is OPT-IN and temporary. The HOT PATH — the one
    # that runs on every env step in a normal run — must contain none.
    _item_lines = [ln for ln in body.splitlines() if ".item()" in ln]
    for ln in _item_lines:
        assert "_ref" in ln or "_d = " in ln, (
            f"a .item() outside the opt-in self-check: {ln.strip()!r} — each "
            f"is a full pipeline drain on every env step")
    n_item = len([ln for ln in _item_lines
                  if not ("_ref" in ln or "_d = " in ln)])
    assert n_item == 0
    print(f"  A3c0. the only .item() left is inside the opt-in A2 self-check "
          f"({len(_item_lines)} line), never on the hot path")
    assert body.count(".tolist()") == 2, (
        "expected exactly one fused transfer per branch (continuous/discrete)")
    print("  A3c. zero .item() drains left in select_action; one fused "
          "transfer per branch (was 3 syncs per step)")


# ---------------------------------------------------------------- A2 -----
def test_A2_carried_features_are_right_or_refused():
    from developmental_ai.policy.actor_critic import StandaloneActorCritic
    from developmental_ai.core.developmental_loop import DevelopmentalAI
    import torch.nn as nn

    torch.manual_seed(0)
    # arch='wm' requires flat CHW pixel obs (3*s*s); 3*4*4 = 48
    enc = nn.Sequential(nn.Linear(48, 12), nn.ELU())
    p = StandaloneActorCritic(obs_dim=48, action_dim=5, hidden_dim=32,
                              continuous=False, arch="wm", enc_dim=12)
    p.attach_shared_encoder(enc)
    rs = np.random.RandomState(0)
    obs_a = rs.rand(48).astype(np.float32)
    obs_b = rs.rand(48).astype(np.float32)

    with torch.no_grad():
        f_a = enc(torch.FloatTensor(obs_a).unsqueeze(0)).reshape(1, -1)
        f_b = enc(torch.FloatTensor(obs_b).unsqueeze(0)).reshape(1, -1)

    # correct pairing: identical result to recomputing, and the self-check
    # agrees exactly (no encoder update fell between)
    a0, i0 = p.select_action(obs_a, deterministic=True)
    a1, i1 = p.select_action(obs_a, deterministic=True, feats=f_a,
                             verify_feats=True)
    assert (a0, i0) == (a1, i1), (a0, i0, a1, i1)
    assert p.last_feat_drift == 0.0
    print(f"  A2. carried features reproduce the recomputed result exactly "
          f"(action {a0}, drift {p.last_feat_drift:.1e})")

    # THE FAILURE THAT MATTERS: another frame's features (a wrong env index,
    # or a frame from before a reset) must RAISE, not silently act
    try:
        p.select_action(obs_a, deterministic=True, feats=f_b,
                        verify_feats=True)
        raise AssertionError(
            "acting on ANOTHER frame's features was accepted — that is the "
            "silent wrong-env / pre-reset bug this check exists to catch")
    except RuntimeError as e:
        assert "do not match" in str(e), e
    print("  A2b. features from a DIFFERENT frame -> RuntimeError, refused "
          "(the wrong-env / pre-reset failure cannot pass silently)")

    # a wrong-width vector is refused even with the check off
    try:
        p.select_action(obs_a, deterministic=True,
                        feats=torch.zeros(1, 7), verify_feats=False)
        raise AssertionError("wrong-width feats accepted")
    except ValueError as e:
        assert "expected (1, 12)" in str(e), e
    print("  A2c. wrong-width feats -> ValueError even with the self-check "
          "off (shape is structural, not a tolerance question)")

    # tolerance admits ONE optimizer step of async encoder drift, and no more
    p.last_feat_drift = 0.0
    p.select_action(obs_a, deterministic=True, feats=f_a + 1e-4,
                    verify_feats=True)
    assert 0 < p.last_feat_drift < 1e-2
    print(f"  A2d. tiny drift ({p.last_feat_drift:.1e}) is tolerated and "
          f"REPORTED — the shared encoder trains asynchronously, so exact "
          f"equality would false-alarm on every WM update")

    # ---- the loop's invalidation contract -----------------------------
    s = types.SimpleNamespace(_reuse_enc_feats=True, _enc_carry=None)
    set_c = DevelopmentalAI._set_enc_carry
    get_f = DevelopmentalAI._feats_for_act

    nxt = [np.zeros(4, dtype=np.float32), np.ones(4, dtype=np.float32)]
    enc_next = torch.arange(8.0).reshape(2, 4)

    # env 0 kept running (same object); env 1 reset (fresh array)
    obs = [nxt[0], np.full(4, 9.0, dtype=np.float32)]
    set_c(s, enc_next, obs, nxt)
    got = get_f(s, 2)
    assert got[0] is not None and torch.equal(got[0], enc_next[0:1])
    assert got[1] is None, (
        "features were carried across a RESET — the policy would act on the "
        "world that existed before the rebuild")
    print("  A2e. identity test: env that kept running carries; env that "
          "RESET gets None (recompute), decided by object identity rather "
          "than by re-deriving `dones[e] or _crashed`")

    # env-count change (client rebuild) invalidates everything
    assert get_f(s, 3) is None
    print("  A2f. env-count change -> full invalidation (no cross-env mixing)")

    # disabled = never carries
    s2 = types.SimpleNamespace(_reuse_enc_feats=False, _enc_carry=None)
    set_c(s2, enc_next, obs, nxt)
    assert s2._enc_carry is None and get_f(s2, 2) is None
    print("  A2g. reuse_encoder_feats false -> no carry at all (clean off)")

    # both stepping bodies must set AND use the carry
    src = open(LOOP).read()
    assert src.count(
        "self._set_enc_carry(encoded_next, obs_list, next_obs_list)") == 2
    assert src.count("feats_per_env=self._feats_for_act(n)") == 2
    print("  A2h. carry set and used in BOTH stepping bodies (2/2)")

    import yaml
    pol = yaml.safe_load(open(os.path.join(
        "configs", "minecraft_skybot.yaml"))).get("policy", {})
    assert pol.get("reuse_encoder_feats") is True
    assert pol.get("verify_encoder_feats") is True, (
        "the self-check must ship ON — it is what turns this from an "
        "optimisation into a proven one; turn it off only after a live run "
        "shows Feat drift ~0")
    print("  A2i. ships with the live self-check ON (costs the saving until "
          "correctness is demonstrated on a real run)")


# ---------------------------------------------------------------- B1 -----
def test_B1_starved_policy_is_reported_as_starved():
    from developmental_ai.policy.actor_critic import StandaloneActorCritic
    p = StandaloneActorCritic(obs_dim=8, action_dim=4, hidden_dim=16,
                              continuous=False)
    for f in ("last_epochs_requested", "last_epochs_capped", "last_rows",
              "last_rows_option", "last_rows_primitive", "last_tau_mean",
              "last_env_steps"):
        assert hasattr(p, f), f"missing update-rate forensic `{f}`"
    print("  B1. update-rate forensics initialised on the policy object")

    # The arithmetic the guard implements, asserted directly: this is the
    # regime the live run is in whenever options are used.
    # NOTE (2026-09-01): this arithmetic now describes the UN-MINIBATCHED
    # path only, which is where its evidence came from — the 2026-07-25
    # collapse was 10 near-identical FULL-BATCH passes over one small,
    # advantage-normalized batch. The live config minibatches, so each step
    # sees a different slice and the bound is `max_updates` plus target_kl
    # instead. Kept as a regression witness for the path it belongs to.
    def epochs(rows, req=10):
        return max(1, min(req, rows // 32))
    assert epochs(51) == 1 and epochs(25) == 1 and epochs(12) == 1
    assert epochs(320) == 10
    print("  B1b. guard arithmetic (full-batch path): 1024 env steps at tau "
          "20/40/80 -> 51/25/12 rows -> 1/1/1 epochs; 320 rows needed for "
          "the configured 10")

    src = open(AC).read()
    assert "PPO STARVED" in src, "the guard must say so at WARNING, not DEBUG"
    assert 'logger.debug(\n                "PPO: %d rows' not in src
    assert '"epochs_capped": bool(' in src, (
        "metrics must carry epochs_capped — without it a starved update is "
        "indistinguishable from KL early-stopping downstream")
    assert '"kl_stopped": bool(' in src and '"max_updates": int(' in src, (
        "with target_kl back on there are THREE causes of a short update "
        "(starved / KL-stopped / budget spent); each must be exported or "
        "they collapse back into one ambiguous 'stopped early'")
    print("  B1c. the guard reports at WARNING and exports `epochs_capped`, "
          "`kl_stopped` and `max_updates`")

    loop = open(LOOP).read()
    assert "<-- STARVED: too few rows for more" in loop
    i_starved = loop.index("<-- STARVED")
    i_kl = loop.index("<-- KL-STOPPED")
    i_good = loop.index("<-- budget: the rollout's gradient-")
    assert i_starved < i_kl < i_good, (
        "the STARVED branch must be tested BEFORE the healthy branches, or a "
        "starved update is again reported as a healthy one")
    assert "PPO update rate:" in loop, (
        "the row composition (option vs primitive, mean tau) must print — it "
        "is what distinguishes 'options ate the rows' from 'little "
        "experience'")
    print("  B1d. the loop distinguishes STARVED from KL-stopped, and prints "
          "the row composition that explains which")

    # ---- B1e, REWRITTEN 2026-09-01 --------------------------------------
    # This used to assert target_kl == 0.0, and said why: with it ON,
    # "stopped early" becomes reachable again "and the two causes must be
    # re-checked, not assumed". target_kl IS on now (0.05), so this discharges
    # that obligation rather than deleting it — the check becomes the thing
    # the old one was protecting: that the causes stay distinguishable.
    #
    # Why turning it on is defensible at all: the 2026-08-05 rollback measured
    # the early stop in the FULL-BATCH path, where an "epoch" is one gradient
    # step over the whole rollout, so the check could only ever fire after the
    # move was already made. The check has always lived inside the minibatch
    # loop, so at minibatch_size 32 it now enforces a budget instead of
    # observing an overshoot. And its second complaint — that stopping left
    # the critic underfit — is fixed directly: the stop freezes the ACTOR and
    # lets the critic train on.
    import yaml
    pol = yaml.safe_load(open(os.path.join(
        "configs", "minecraft_skybot.yaml"))).get("policy", {})
    _kl_cfg = float(pol.get("target_kl", 0.0))
    _mb_cfg = int(pol.get("minibatch_size", 0) or 0)
    if _kl_cfg > 0.0:
        assert _mb_cfg > 0, (
            f"target_kl={_kl_cfg} with minibatch_size=0 recreates the exact "
            f"2026-08-05 failure: an 'epoch' is then ONE full-batch step, so "
            f"the KL check always overshoots its budget instead of enforcing "
            f"it. Enable minibatching or turn target_kl back off.")
        assert "_actor_frozen" in src, (
            "target_kl is on but the stop is not actor-only — the 2026-08-05 "
            "rollback's second finding was that abandoning the update leaves "
            "the critic underfit, which makes the NEXT update worse")
        assert "kl_stopped" in loop, (
            "target_kl is on and the loop cannot report it — 'stopped early' "
            "is reachable again and must be named, not inferred")
    print(f"  B1e. target_kl={_kl_cfg} with minibatch_size={_mb_cfg}: the "
          f"stop is per-minibatch (enforces a budget, not observes an "
          f"overshoot), actor-only (critic keeps fitting), and named in the "
          f"log — the three conditions the 2026-08-05 rollback required")


if __name__ == "__main__":
    print("A1. curiosity training batches (same gradients, fewer launches)")
    test_A1_curiosity_batching()
    test_A1_no_duplicated_body_drift()
    print("A3. one device->host transfer per action selection")
    test_A3_single_sync_in_select_action()
    print("A2. the carried encoder forward is right, or refused")
    test_A2_carried_features_are_right_or_refused()
    print("B1. a starved policy reports as starved")
    test_B1_starved_policy_is_reported_as_starved()
    print("[throughput-wave] ALL PASS")
