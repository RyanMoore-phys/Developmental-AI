"""Perspective-from-motion smoke (2026-09-18).

WHAT THIS WAVE CHANGED, AND WHY IT NEEDS ITS OWN GUARDS

    SkyBot saw a flat 128x128 frame and had never been given a reason to
    recover the 3D world behind it. The scoreboard says how that went:
    13,305 blocks broken for 35 logs, every log skill at 0/20, and a history
    of income earned by staring at the sky (96% of the drive), sitting in a
    menu (77% of income) and holding attack at an unreachable trunk (96% of
    option activity, 0 logs).

    The fix is a FLOW HEAD: from the latent and an action, predict the dense
    displacement field that action would produce, and train it by whether
    that field warps this frame into the next one. Under a forward step,
    predicted flow magnitude IS inverse depth. Nothing labels the flow.

    THE DIAGNOSIS THAT WAS WRONG FIRST TIME, recorded so nobody re-derives
    it. The claim "the encoder throws away spatial identity at the
    bottleneck" is FALSE: CNNEncoder does reshape -> Linear, and a
    flatten+Linear is position-specific. The real defects were three, and
    only measurement separated them:
      1. the bottleneck was 4x4, so one cell covered a 32x32 px slab;
      2. flat per-pixel MSE makes a three-block trunk rounding error against
         sky, ground and texture;
      3. nothing downstream ever indexed position anyway.

    EVERY SENSE THIS ADDS IS UNPAID. They enter proprioception only, on the
    doctrine _reach_sense states. The one thing that touches the reward path
    is the flow residual joining LEARNING PROGRESS as a second error channel
    — and LP pays for error going DOWN, so that changes what the agent can
    make progress on, not what it is paid for standing still.

Contracts:
    A. ZERO-MOTION, ZERO INCOME. Two identical frames -> residual exactly 0.
       This is the property the whole design rests on: no ego-motion, no
       predicted flow, nothing to be wrong about, no income. Structural, not
       a guard.
    B. CONSTANT-SENSE WAGE TEST. Holding every new sense pinned must pay
       exactly 0 over N steps. Keeps the flawed telescoping form alongside as
       a regression witness (CLAUDE.md 4.3: a potential written the textbook
       way pays -w(1-gamma)Phi every step Phi is held).
    C. SKY / APERTURE. A texture-free frame makes the photometric term
       satisfiable by ANY flow. The smoothness prior must keep the residual
       at ~0 rather than letting the head manufacture one.
    D. RESET SPLICE. A stream whose previous frame belongs to a different
       world must report 0, not a near-total residual that reads as the most
       interesting event in the run.
    E. DUPLICATED BODY (CLAUDE.md 4.2). The senses must be computed in BOTH
       loop bodies, and the proprio width the policy declares must match what
       _augment_proprio writes.
    F. POSITION IS REPRESENTABLE. The same patch at two image positions must
       produce different encoder output.
    G. DEPTH ORDERING — the real one. On a synthetic two-plane scene, after
       training, the NEAR plane's predicted flow must exceed the FAR plane's.
       If this fails, nothing downstream is worth building.

Run: PYTHONPATH=. python tests/_perspective_smoke.py
"""
import os
import sys

sys.path.insert(0, ".")

import math

import numpy as np
import torch
import torch.nn.functional as F

from developmental_ai.world_model.rssm import (
    CNNEncoder, FlowHead, WorldModel, flow_warp, charbonnier, photometric,
    ego_from_proprio, flow_from_depth, _cnn_channel_ladder,
)

SRC = os.path.join("developmental_ai", "core", "developmental_loop.py")
CFG = os.path.join("configs", "minecraft_skybot.yaml")

IMG = 32          # small frames: every contract here is about SHAPE of the
                  # signal, not about pixel fidelity, and 32px keeps the
                  # whole file runnable on a laptop in seconds.


def _wm(**kw):
    """A tiny pixel world model with the flow head on."""
    d = dict(obs_dim=3 * IMG * IMG, action_dim=4, stochastic_size=4,
             stochastic_classes=4, deterministic_size=32, hidden_dim=32,
             pixel_obs=True, image_size=IMG, film_conditioning=True,
             min_grid=4, coord_channels=True, readout_channels=16,
             flow_head=True, flow_size=16, flow_photo_size=IMG)
    d.update(kw)
    return WorldModel(**d)


def test_zero_motion_zero_income():
    """A. Nothing moved -> nothing to explain -> residual EXACTLY zero.

    Not "small". Zero. The residual is a ratio whose denominator is how much
    the frame changed at all, and `motion_floor` returns a hard 0 below the
    noise of an unchanged image. An agent that stands still earns nothing
    from this channel by construction, which is the one property that makes
    it safe to attach to the drive at all.
    """
    wm = _wm()
    frame = torch.rand(1, 3 * IMG * IMG)
    lat = torch.randn(1, wm.rssm.latent_dim)
    act = F.one_hot(torch.tensor([1]), 4).float()

    r = wm.flow_residual(frame, frame.clone(), lat, act)
    assert float(r[0]) == 0.0, f"identical frames paid {float(r[0])}"

    # And a nearly-identical frame (sensor-grade jitter) is still zero.
    jitter = (frame + torch.randn_like(frame) * 1e-4).clamp(0, 1)
    r2 = wm.flow_residual(frame, jitter, lat, act)
    assert float(r2[0]) == 0.0, f"jitter-only frames paid {float(r2[0])}"

    # A REAL change does produce a signal, or the test above proves nothing.
    moved = torch.roll(frame.view(1, 3, IMG, IMG), 5, dims=3).reshape(1, -1)
    r3 = wm.flow_residual(frame, moved, lat, act)
    assert float(r3[0]) > 0.0, "a genuinely moved frame must not read as 0"
    print(f"  A. still={float(r[0]):.6f} jitter={float(r2[0]):.6f} "
          f"moved={float(r3[0]):.4f} — no motion, no income")


def test_constant_sense_pays_nothing():
    """B. A pinned sense must pay 0/step, and the telescoping form must not.

    CLAUDE.md 4.3, kept as a live witness rather than prose. `F = w(g*P' - P)`
    pays `-w(1-g)P` on EVERY step P is held constant — for a cost potential
    (P < 0) that is a WAGE for sitting still, which shipped once and showed
    up in the log as "Gaze level: +0.00014/step".

    The flow senses are not potentials at all — they are proprioception, and
    proprioception pays nothing. This asserts the arithmetic that would have
    to hold IF anyone ever turns one into a shaped term.
    """
    g, w, steps = 0.99, 1.0, 100
    phi = -0.5                            # a COST potential, held pinned

    telescoping = sum(w * (g * phi - phi) for _ in range(steps))
    plain = sum(w * (phi - phi) for _ in range(steps))

    assert abs(plain) < 1e-12, f"plain difference paid {plain}"
    assert telescoping > 1e-3, (
        "the flawed form must still demonstrate the defect, or this witness "
        "has stopped witnessing anything")
    print(f"  B. pinned over {steps} steps: plain={plain:+.6f}  "
          f"telescoping={telescoping:+.6f} (the known defect, kept as witness)")


def test_sky_aperture_pays_nothing():
    """C. A texture-free frame must not manufacture a residual.

    THE APERTURE PROBLEM IS AN EXPLOIT SURFACE. Over a flat region the
    photometric term is satisfied by ANY displacement — the field is
    unidentifiable — so a head with no prior would emit arbitrary values and
    produce residual out of nothing. That is precisely how this project's
    previous drives ended up paying 96% of the agent's income for staring at
    the sky, one mechanism over.

    Sky is also the case where the DENOMINATOR is ~0: a uniform frame that
    pans is still a uniform frame. So the residual is zero for the same
    reason standing still is.
    """
    wm = _wm()
    sky = torch.full((1, 3 * IMG * IMG), 0.62)         # flat blue-grey
    lat = torch.randn(1, wm.rssm.latent_dim)
    act = F.one_hot(torch.tensor([1]), 4).float()

    panned = torch.roll(sky.view(1, 3, IMG, IMG), 6, dims=3).reshape(1, -1)
    r = wm.flow_residual(sky, panned, lat, act)
    assert float(r[0]) == 0.0, (
        f"a panning flat sky paid {float(r[0])} — the aperture problem is "
        f"being farmed")
    print(f"  C. flat sky panned 6px: residual={float(r[0]):.6f} — "
          f"an unidentifiable field pays nothing")


def test_reset_splice_pays_nothing():
    """D. A frame pair spanning two different worlds must read 0.

    Every episode boundary, death and client rebuild splices two unrelated
    frames across one index. Without the `valid` mask the warp fails totally
    there and the residual reads ~1.0 — so the single most 'interesting'
    thing in any run would be the resets, and curiosity would chase them.
    """
    wm = _wm()
    a = torch.rand(2, 3 * IMG * IMG)
    b = torch.rand(2, 3 * IMG * IMG)          # an unrelated world
    lat = torch.randn(2, wm.rssm.latent_dim)
    act = F.one_hot(torch.tensor([1, 1]), 4).float()

    unmasked = wm.flow_residual(a, b, lat, act)
    assert float(unmasked[0]) > 0.1, (
        "the splice must genuinely look unexplained, or the mask below is "
        "not actually protecting anything")

    valid = torch.tensor([False, True])
    masked = wm.flow_residual(a, b, lat, act, valid=valid)
    assert float(masked[0]) == 0.0, f"invalid stream paid {float(masked[0])}"
    assert float(masked[1]) == float(unmasked[1]), (
        "masking one stream must not disturb another")
    print(f"  D. splice unmasked={float(unmasked[0]):.3f} -> "
          f"masked={float(masked[0]):.3f} — a reset is not a discovery")


def test_source_contracts():
    """E. Both loop bodies, and a proprio width that matches its writer.

    CLAUDE.md 4.2: developmental_loop.py steps the environment in TWO places
    and an edit that lands in one is silent. This project has already lost
    the reach sense that way — it was appended in the episodic body only and
    was dead in every skybot run.
    """
    src = open(SRC).read()

    n_sense = src.count("self._flow_now = self._flow_senses(")
    assert n_sense == 2, (
        f"flow senses computed in {n_sense} loop bodies, expected 2 — "
        f"_run_episode_parallel and _collect_segment (CLAUDE.md 4.2)")

    n_resid = src.count("_fr = self._flow_residual_batch(")
    assert n_resid == 2, (
        f"flow residual computed in {n_resid} loop bodies, expected 2")

    n_cache = src.count("self._flow_prev_latent = latent.detach()")
    assert n_cache == 2, (
        f"the one-step latent memory is refreshed in {n_cache} bodies, "
        f"expected 2 — a stale cache silently scores the wrong transition")

    # The residual must be measured on the SAME transition curiosity scores.
    assert "extra_error=_fr" in src, (
        "the flow residual must reach compute_intrinsic_reward as an extra "
        "ERROR channel; anything else would be a new income stream")

    # WIDTH. _augment_proprio writes 4 symbolizer fields + 4 flow fields;
    # the policy must declare exactly that.
    assert src.count('"flow_fovea"') >= 1 and src.count('"mover"') >= 1
    assert "ext = [0.0, 0.0, 0.0, 0.0] if self.symbolizer is not None else []" in src, (
        "the four symbolizer-derived proprio fields must be conditional on "
        "the symbolizer, or the vector is four wider than the policy expects "
        "in a symbolizer-less run and every flow sense is read from the "
        "wrong slot")

    # The fovea head must not read rssm.obs_dim as 'the encoder width' —
    # proprio-in-the-embed broke that identity.
    assert "vec_dim" in src, (
        "the fovea head's input width must come from encoder.vec_dim; "
        "rssm.obs_dim now includes proprioception")
    print(f"  E. both bodies: senses={n_sense} residual={n_resid} "
          f"cache={n_cache}; proprio width conditional; fovea reads vec_dim")


def test_position_is_representable():
    """F. The encoder must distinguish a patch by WHERE it is.

    Not a tautology worth skipping: it is the one thing the whole spatial
    half of this wave claims to buy, and coord_channels + an 8x8 grid are
    cheap enough that a silent misconfiguration (min_grid back at 4, coords
    off) would go unnoticed for weeks.
    """
    assert _cnn_channel_ladder(128) == [32, 64, 128, 256, 256]
    assert _cnn_channel_ladder(128, min_grid=8) == [32, 64, 128, 256]

    enc = CNNEncoder(3, 64, IMG, min_grid=8, coord_channels=True,
                     readout_channels=16)
    assert enc.grid == 8, f"expected an 8x8 map, got {enc.grid}x{enc.grid}"
    assert enc.vec_dim == 64

    patch = torch.rand(3, 8, 8)
    left = torch.zeros(1, 3, IMG, IMG); left[0, :, 4:12, 4:12] = patch
    right = torch.zeros(1, 3, IMG, IMG); right[0, :, 4:12, 20:28] = patch

    vl, ml = enc.encode_with_map(left.reshape(1, -1))
    vr, mr = enc.encode_with_map(right.reshape(1, -1))
    dv = float((vl - vr).abs().mean())
    dm = float((ml - mr).abs().mean())
    assert dv > 1e-4, f"same patch at two positions gave identical vectors ({dv})"
    assert dm > 1e-4, f"same patch at two positions gave identical maps ({dm})"
    assert ml.shape[-1] == 8
    print(f"  F. patch moved 16px: vector delta={dv:.5f} map delta={dm:.5f} "
          f"on an {enc.grid}x{enc.grid} grid")


def test_depth_ordering_is_learned():
    """G. THE REAL ONE. Near must sweep faster than far, learned from pixels.

    A synthetic two-plane scene: the left half shifts 4 px per step (near),
    the right half 1 px (far). Nothing tells the head which is which — the
    only signal is whether its predicted field warps frame t into frame t+1.

    If this passes, the mechanism the whole wave rests on works: the
    counterfactual "how fast would this sweep past me if I stepped forward"
    is inverse depth, and it is available WITHOUT walking. If it fails,
    nothing downstream — the senses, the residual channel, the object slots
    — is worth building.
    """
    torch.manual_seed(0)
    S, N = 32, 8
    base = torch.rand(1, 3, S, S)

    def scene(near_shift, far_shift):
        a = torch.roll(base, shifts=near_shift, dims=3)
        b = torch.roll(base, shifts=far_shift, dims=3)
        out = base.clone()
        out[:, :, :, :S // 2] = a[:, :, :, :S // 2]
        out[:, :, :, S // 2:] = b[:, :, :, S // 2:]
        return out

    src = torch.cat([scene(0, 0)] * N)
    tgt = torch.cat([scene(-4, -1)] * N)
    lat = torch.randn(N, 16)
    act = F.one_hot(torch.zeros(N, dtype=torch.long), 3).float()

    head = FlowHead(16, 3, out_size=S, hidden_dim=64, base_channels=32)
    # ZERO-INIT CONTRACT: the head starts predicting "the world is static",
    # which is the right prior and keeps the first thousand warps from
    # scrambling frames with random resampling.
    assert float(head(lat, act).abs().max()) == 0.0, (
        "FlowHead must start at exactly zero flow")

    opt = torch.optim.Adam(head.parameters(), lr=3e-3)
    first = None
    for _ in range(400):
        loss = charbonnier(flow_warp(src, head(lat, act)) - tgt).mean()
        first = first if first is not None else float(loss)
        opt.zero_grad(); loss.backward(); opt.step()

    assert float(loss) < first, (
        f"photometric loss did not fall ({first:.5f} -> {float(loss):.5f})")

    mag = head(lat, act).detach().norm(dim=1)
    near = float(mag[:, :, :S // 2].mean())
    far = float(mag[:, :, S // 2:].mean())
    # 1.1x, not the measured 1.20x: the margin is the CONTRACT (near must
    # sweep faster than far), the measurement is a report. Pinning the exact
    # value would make every legitimate change to the warp look like a
    # regression — and fixing the align_corners bug legitimately moved it
    # from 1.40x to 1.20x by removing a blur floor that was inflating both.
    assert near > far * 1.1, (
        f"depth ordering NOT recovered: near={near:.4f} far={far:.4f}. The "
        f"near plane moved 4x as fast and must predict a larger field.")
    print(f"  G. photometric {first:.5f} -> {float(loss):.5f}; "
          f"near={near:.4f} far={far:.4f} ({near / max(far, 1e-9):.2f}x) — "
          f"inverse depth learned from motion alone")


def test_surprise_map_localizes():
    """R. P4 — WHERE the model is wrong, not just how much.

    The per-pixel warp error was already being computed inside
    flow_residual and then averaged into one scalar. Curiosity has therefore
    been SCENE-LEVEL since the project began: the agent could be surprised,
    but never surprised BY A PLACE IN THE FRAME — which is why the vision
    magnet can want a category and never an instance.

    Same scale-free ratio per cell as the scalar uses globally: local warp
    error over local frame change. A cell where nothing moved reads 0, for
    exactly the reason standing still pays nothing.
    """
    torch.manual_seed(0)
    wm = _wm()
    lat = torch.randn(1, wm.rssm.latent_dim)
    act = F.one_hot(torch.tensor([1]), 4).float()

    base = torch.rand(1, 3, IMG, IMG)
    moved = base.clone()
    q = IMG // 4
    moved[:, :, :q, :q] = torch.rand(1, 3, q, q)      # only the top-left moves

    r, m = wm.flow_residual(base.reshape(1, -1), moved.reshape(1, -1),
                            lat, act, return_map=True, map_size=8)
    assert m.shape == (1, 1, 8, 8)

    hot = float(m[0, 0, :2, :2].mean())
    cold = float(m[0, 0, 4:, 4:].mean())
    assert hot > cold + 0.3, (
        f"the changed quadrant reads {hot:.2f} and the untouched one "
        f"{cold:.2f}; surprise is not localizing")
    assert cold < 0.1, (
        f"an untouched region reads {cold:.2f}; a cell where nothing moved "
        f"must read ~0 or the map manufactures curiosity out of stillness")

    # And a wholly static pair is zero everywhere — the P4 form of contract A.
    _r0, m0 = wm.flow_residual(base.reshape(1, -1), base.reshape(1, -1),
                               lat, act, return_map=True, map_size=8)
    assert float(m0.abs().max()) == 0.0, (
        "a static frame produced non-zero surprise somewhere")
    print(f"  R. changed quadrant {hot:.2f} vs untouched {cold:.2f}; static "
          f"frame max {float(m0.abs().max()):.3f}")


def test_horizon_term_and_the_floor_that_hid_it():
    """S. P1 — supervising the horizon the dreams actually live at.

    The world model is trained ONE step ahead while the policy trains on
    imagined rollouts fifteen steps long, so the horizon that matters has
    never been supervised. Each entry in `latent_horizons` rolls the prior
    forward k steps with the actions actually taken and charges the
    divergence from the posterior that resulted.

    THE INSTRUMENT HAD TO BE FIXED FIRST, and that is worth recording.
    `compute_kl_loss` applies free nats of 0.1 to BOTH halves, so its value
    cannot fall below 1.0*0.1 + 0.1*0.1 = 0.11 and its gradient vanishes
    underneath. MEASURED: two models with completely different training both
    reported 0.1100 on a 16-step rollout across three seeds — the
    measurement was pinned at its own floor and could see nothing. Free nats
    exist to stop the posterior collapsing toward the prior early on; that
    is not the concern sixteen steps out. So the horizon term uses the raw
    divergence and `compute_kl_loss` keeps its floor for the one-step term
    it was tuned for.

    MEASURED BENEFIT on a deterministic toy world, 16-step rollout
    divergence after 150 steps: 0.066 / 0.065 / 0.030 one-step-only versus
    0.0002 with horizons [4, 16] — every seed, two orders of magnitude.
    """
    wm = _wm()
    b = 4
    pri = torch.randn(b, wm.rssm.stoch_dim)
    same = pri.clone()

    floor = float(wm.rssm.compute_kl_loss(pri, same))
    raw = float(wm.rssm.prior_divergence(pri, same))
    assert abs(floor - 0.11) < 1e-4, (
        f"the free-nats floor is {floor:.4f}, not 0.11 — the docstring above "
        f"records a measurement that no longer holds")
    assert raw < 1e-5, (
        f"an IDENTICAL prior and posterior diverge by {raw:.6f}; the raw "
        f"measure must read zero when the rollout is exactly right, or it "
        f"has a floor of its own")

    # Off by default and reported; on, it contributes.
    off = _wm(latent_horizons=[])
    on = _wm(latent_horizons=[4])
    assert off.latent_horizons == [] and on.latent_horizons == [4]
    # Horizons shorter than 2, or longer than the sequence, are inert rather
    # than an error — a config must not be able to crash a training block.
    assert _wm(latent_horizons=[1, 0, -3]).latent_horizons == []

    B, L = 2, 12
    obs = torch.rand(B, L, 3 * IMG * IMG)
    act = F.one_hot(torch.randint(0, 4, (B, L)), 4).float()
    rew, cont = torch.zeros(B, L), torch.ones(B, L)

    torch.manual_seed(5)
    lo = off.compute_loss(obs, act, rew, cont)
    torch.manual_seed(5)
    lh = on.compute_loss(obs, act, rew, cont)
    assert float(lo["horizon"]) == 0.0, (
        "an empty horizon list must contribute exactly nothing")
    assert float(lh["horizon"]) > 0.0, "the horizon term is inert when enabled"

    # A horizon longer than the sequence cannot produce starts, so it must
    # degrade to zero rather than indexing past the end.
    torch.manual_seed(5)
    lng = _wm(latent_horizons=[64]).compute_loss(obs, act, rew, cont)
    assert float(lng["horizon"]) == 0.0
    print(f"  S. free-nats floor {floor:.4f} (raw {raw:.2e}); horizon "
          f"0 when off, {float(lh['horizon']):.4f} when on, 0 when k > seq_len")


def test_buffer_proprio_column():
    """I. The body column must be optional, aligned, and never load misaligned.

    THE PRECEDENT THIS FOLLOWS IS `restarts`. That column arrived 2026-09-01
    against a persisted buffer that predated it, and was made optional on load
    for a reason worth restating: the pod's buffer is the ONLY copy of the
    agent's experience, and a new column that refuses to load bricks it. A pod
    has already died with ~11 days of unbacked state.

    The width check is the sharper hazard. If PROPRIO_KEYS ever changes, field
    k means a different sense than it did, and silently carrying the old values
    over would teach the model a body it does not have — a failure that looks
    like nothing at all until the flow head cannot learn.
    """
    import tempfile
    from developmental_ai.world_model.replay_buffer import (
        ReplayBuffer, MultiStreamReplayBuffer)

    OD, AD, PD = 12, 3, 5

    # Every non-Minecraft config: no column at all, no key in the batch.
    b0 = ReplayBuffer(capacity=200, obs_dim=OD, action_dim=AD)
    for i in range(60):
        b0.add(np.full(OD, i / 100., np.float32), 1, 0.0, False)
    assert b0.proprio is None
    assert "proprio" not in b0.sample_sequences(4, 8), (
        "a buffer with no body column must not emit a `proprio` key — a "
        "zero tensor would look like a body that reads flat")

    b = ReplayBuffer(capacity=200, obs_dim=OD, action_dim=AD, proprio_dim=PD)
    for i in range(60):
        b.add(np.full(OD, i / 100., np.float32), 1, 0.0, False,
              proprio=np.full(PD, i / 100., np.float32))
    batch = b.sample_sequences(4, 8)
    assert batch["proprio"].shape == (4, 8, PD)
    i0 = int(batch["start_indices"][0])
    assert abs(float(batch["proprio"][0, 0, 0]) - i0 / 100.) < 1e-6, (
        "the body column must be gathered at the SAME indices as the frames")

    # A stream with no body sense writes neutral zeros rather than raising.
    b.add(np.zeros(OD, np.float32), 0, 0.0, False, proprio=None)
    assert float(np.abs(b.proprio[b.position - 1]).sum()) == 0.0

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "buf")
        b.save(p)
        b2 = ReplayBuffer(capacity=200, obs_dim=OD, action_dim=AD,
                          proprio_dim=PD)
        b2.load(p)
        assert np.allclose(b2.proprio[:b2.size], b.proprio[:b.size])

        os.remove(os.path.join(p, "proprio.npy"))     # a PRE-WAVE buffer
        b3 = ReplayBuffer(capacity=200, obs_dim=OD, action_dim=AD,
                          proprio_dim=PD)
        n3 = b3.load(p)
        assert n3 > 0 and float(np.abs(b3.proprio[:n3]).sum()) == 0.0, (
            "a buffer saved before this wave must load as neutral zeros, not "
            "refuse — it is the only copy of the agent's experience")

        b.save(p)
        b4 = ReplayBuffer(capacity=200, obs_dim=OD, action_dim=AD,
                          proprio_dim=PD + 2)          # PROPRIO_KEYS changed
        n4 = b4.load(p)
        assert float(np.abs(b4.proprio[:n4]).sum()) == 0.0, (
            "a width change must restore NEUTRAL ZEROS: field k now means a "
            "different sense and carrying the old values over teaches a body "
            "the agent does not have")

    m = MultiStreamReplayBuffer(num_streams=2, capacity=400, obs_dim=OD,
                                action_dim=AD, proprio_dim=PD)
    for st in (0, 1):
        for i in range(60):
            m.add(np.full(OD, i / 100., np.float32), 1, 0.0, False, stream=st,
                  proprio=np.full(PD, (st + 1) / 10., np.float32))
    assert m.sample_sequences(4, 8)["proprio"].shape == (4, 8, PD)

    g = ReplayBuffer(capacity=5000, obs_dim=OD, action_dim=AD, proprio_dim=PD,
                     growth={"enabled": True, "block_transitions": 1000})
    cap0 = g.capacity
    for _ in range(1200):
        g.add(np.zeros(OD, np.float32), 0, 0.0, False,
              proprio=np.full(PD, 0.5, np.float32))
        g.maybe_grow()
    assert g.capacity > cap0 and len(g.proprio) == g.capacity, (
        "growth must publish the body block with every other column, or a "
        "grown buffer indexes past the end of it")
    print(f"  I. body column: optional, index-aligned, width-guarded, "
          f"multi-stream, grows {cap0}->{g.capacity}")


# ===========================================================================
# WAVE 2 (2026-09-18) — a correct photometric loss, and one measured retreat
# ===========================================================================

def test_zero_flow_is_the_identity_warp():
    """J. THE WAVE 1 BUG. warp(img, 0) must return img, to float precision.

    `flow_warp` shipped with `align_corners=False` against a
    linspace(-1, 1, n) base grid. Those two conventions do not pair: under
    align_corners=False, -1 and +1 are the OUTER EDGES of the corner pixels
    rather than their centres, so the identity grid resamples at a half-pixel
    offset.

    MEASURED on a random 16x16 image: max |warp(img, 0) - img| was 0.490 with
    False and 0.000001 with True.

    WHAT IT COST. Zero flow was not the identity, so the photometric loss
    charged a floor of blur error that NO field could remove, and the head's
    cheapest route to reducing it was a constant compensating offset — a bias
    on every depth reading. The residual inflated the same way, since its
    denominator uses no warp at all.

    WHY CONTRACT A DID NOT CATCH IT: the zero-motion test gates on
    `motion_floor` before the warp runs, so identical frames returned 0 by the
    gate and never exercised this path. A guard hid a bug from the test whose
    job was to find it, which is worth a contract of its own.
    """
    torch.manual_seed(0)
    img = torch.rand(2, 3, 16, 16)
    out = flow_warp(img, torch.zeros(2, 2, 16, 16))
    err = float((out - img).abs().max())
    assert err < 1e-5, (
        f"zero flow is not the identity warp (max err {err:.6f}) — the loss "
        f"is charging a blur floor no field can remove")

    # And the helper agrees: identity error against the SAME frame is ~0.
    we, ie = photometric(img, img, torch.zeros(2, 2, 16, 16))
    assert float(we.max()) < 2e-3 and float(ie.max()) < 2e-3
    print(f"  J. warp(img, 0) max err {err:.8f} — zero flow IS the identity")


def test_automask_masks_the_right_pixels():
    """K. Auto-masking keeps what the warp explains and starts at 100%.

    A pixel whose error is WORSE after warping is better explained by
    "nothing moved" — occluded, or texture travelling with the camera.
    Charging it supervises the field toward a wrong answer, and those pixels
    cluster at depth discontinuities: trunk edges.

    THE LATCH THIS AVOIDS. At init the head emits zero flow, so warp error
    EQUALS identity error everywhere. A strict `<` would mask every pixel,
    make the loss identically zero, and leave the head with no gradient to
    escape with — guard-becomes-latch (CLAUDE.md 4.1) in the tensor math.
    `<=` plus a tie epsilon means the mask starts at 100% retained.

    MEASURED BENEFIT, on a two-plane scene carrying a genuine occluder:
    near/far ratio 1.11/1.18/1.15x unmasked -> 1.59/1.51/1.64x masked across
    three seeds, 87% of pixels retained. Wins every seed.
    """
    torch.manual_seed(0)
    src = torch.rand(1, 3, 16, 16)
    tgt = torch.roll(src, 2, dims=3)

    we, ie = photometric(src, tgt, torch.zeros(1, 2, 16, 16))
    keep = (we <= ie + 1e-6)
    assert bool(keep.all()), (
        "at zero flow the auto-mask must retain EVERY pixel — a strict '<' "
        "here zeroes the loss and the head can never produce a gradient")

    # With a real field, the mask must be exactly the agreed predicate.
    flow = torch.randn(1, 2, 16, 16) * 0.05
    we, ie = photometric(src, tgt, flow)
    keep = (we <= ie + 1e-6)
    expected = (we <= ie + 1e-6)
    assert bool((keep == expected).all())
    frac = float(keep.float().mean())
    assert 0.0 < frac <= 1.0
    print(f"  K. mask retains 100% at zero flow; {frac:.0%} under a random "
          f"field — masked pixels are exactly those the warp makes worse")


def test_multiscale_is_structural():
    """L. Every requested scale contributes, and the loss changes.

    HONEST SCOPE. I will not assert "the pyramid always wins": measured on a
    32px synthetic it beat single-scale in 5 of 6 (shift, seed) pairs and LOST
    one. That is a real but weak effect and a flaky contract. What IS
    assertable is the mechanism — that the scales are actually evaluated and
    that [1] reproduces the single-scale loss exactly.
    """
    torch.manual_seed(0)
    src = torch.rand(2, 3, 32, 32)
    tgt = torch.roll(src, 3, dims=3)
    flow = torch.randn(2, 2, 32, 32) * 0.05

    single = float(photometric(src, tgt, flow)[0].mean())
    terms = []
    for sc in (1, 2, 4):
        ps = max(8, 32 // sc)
        a = src if ps == 32 else F.interpolate(src, size=(ps, ps), mode="area")
        b = tgt if ps == 32 else F.interpolate(tgt, size=(ps, ps), mode="area")
        f = flow if ps == 32 else F.interpolate(
            flow, size=(ps, ps), mode="bilinear", align_corners=True)
        terms.append(float(photometric(a, b, f)[0].mean()))
    assert abs(terms[0] - single) < 1e-6, "scale 1 must be the single-scale term"
    assert len(set(round(t, 6) for t in terms)) > 1, (
        "the coarse scales are not contributing anything distinct")
    print(f"  L. scale terms {[round(t, 4) for t in terms]} — scale 1 is "
          f"exactly single-scale, coarse scales differ")


def test_longer_baselines_are_off_and_why():
    """M. THE MEASURED RETREAT, kept as a witness rather than deleted.

    I recommended longer baselines on classical-SfM reasoning: depth
    resolution scales with baseline, adjacent frames are the worst baseline
    available, so warping t->t+k for k in {1,2,4} from the same head should
    sharpen depth for free.

    IT DOES THE OPPOSITE. On the two-plane synthetic (near band moving 4x the
    far, truth 4.0x), across three seeds:

        strides [1]      1.60x  1.28x  1.06x
        strides [1,2,4]  0.86x  0.79x  0.84x   <- ORDERING INVERTED

    Below 1.0 the head reports the NEAR plane as moving slower than the far
    one — worse than no depth signal at all.

    DIAGNOSIS, mechanical rather than a toy artifact: one head emits one field
    per (latent, action), and at stride k the true displacement is k times
    larger while `max_flow` bounds the field. A long-stride term can be
    literally unrepresentable, and an unlearnable term does not sit quietly —
    it drags the shared head into a compromise that destroys the stride-1
    field the agent acts on.

    THE FIX WAS TRIED. Scaling the bound with the baseline (range_scale,
    kept because it is correct regardless) gave 0.79x / 1.16x / 1.46x: better
    than broken, still worse than stride [1] on two of three seeds.

    This contract pins the REPRESENTABILITY relation that explains it, and
    pins the live config to strides [1] so nobody turns this back on without
    reading the above.
    """
    h = FlowHead(8, 3, out_size=16, hidden_dim=32, base_channels=16)
    lat, act = torch.randn(1, 8), torch.zeros(1, 3)
    act[0, 0] = 1.0
    with torch.no_grad():
        torch.nn.init.normal_(h.out.weight, std=1.0)
        torch.nn.init.normal_(h.out.bias, std=1.0)
        cap1 = float(h(lat, act, range_scale=1.0).abs().max())
        cap4 = float(h(lat, act, range_scale=4.0).abs().max())
    assert cap4 > cap1 * 3.0, (
        f"range_scale must widen the representable field with the baseline "
        f"({cap1:.3f} -> {cap4:.3f}); without it a stride-4 displacement "
        f"cannot be expressed and poisons the shared head")

    import yaml
    cfg = yaml.safe_load(open(CFG))
    strides = cfg.get("world_model", {}).get("flow_strides", [1])
    assert list(strides) == [1], (
        f"flow_strides is {strides}, but longer baselines were MEASURED to "
        f"invert depth ordering (see this docstring). Turning them on needs a "
        f"head conditioned on the stride as a first-class input.")
    print(f"  M. range_scale widens the field {cap1:.3f} -> {cap4:.3f}; "
          f"live config pinned to strides {list(strides)} (measured retreat)")


def test_ego_from_proprio_is_sign_safe():
    """N. Body motion from stored proprioception, with the sign pinned.

    WHY A KNOWN-ANSWER TEST AND NOT A DERIVATION. Minecraft yaw grows
    CLOCKWISE while atan2 grows counter-clockwise, and this repo has already
    shipped exactly that sign error once — the episodic bearing in
    _augment_proprio read "dead ahead" as "behind", and the yaw=0 case cannot
    see it. So the convention gets pinned by cases, not by argument.

    Yaw also must come from the (sin, cos) PAIR rather than a subtraction of
    angles: heading is stored as a pair precisely because it wraps, and a
    difference of atan2s jumps by 2*pi once per revolution.
    """
    layout = {"pitch": 9, "moved": 10, "head_sin": 11, "head_cos": 12}
    D = 13

    def row(yaw_rad, pitch=0.5, moved=0.0):
        v = torch.zeros(1, D)
        v[0, layout["pitch"]] = pitch
        v[0, layout["moved"]] = moved
        v[0, layout["head_sin"]] = math.sin(yaw_rad)
        v[0, layout["head_cos"]] = math.cos(yaw_rad)
        return v

    # A quarter turn, the easy case.
    e = ego_from_proprio(row(0.0), row(math.pi / 2), layout)
    assert abs(float(e[0, 0]) - math.pi / 2) < 1e-5, float(e[0, 0])

    # THE WRAP CASE, which a subtraction of angles fails: 350deg -> 10deg is a
    # +20deg turn, not -340deg.
    e = ego_from_proprio(row(math.radians(350)), row(math.radians(10)), layout)
    assert abs(float(e[0, 0]) - math.radians(20)) < 1e-5, (
        f"wrap-around yaw gave {math.degrees(float(e[0, 0])):.1f} deg, "
        f"expected +20 — heading must come from the sin/cos pair")

    # Sign: the opposite turn must be the opposite sign.
    e2 = ego_from_proprio(row(math.radians(10)), row(math.radians(350)), layout)
    assert float(e2[0, 0]) < 0 < float(e[0, 0])

    # Pitch is a plain difference; forward distance is `moved` scaled.
    e3 = ego_from_proprio(row(0.0, pitch=0.4), row(0.0, pitch=0.6, moved=0.5),
                          layout, move_scale=2.0)
    assert abs(float(e3[0, 1]) - 0.2 * math.pi) < 1e-5
    assert abs(float(e3[0, 2]) - 1.0) < 1e-5
    print("  N. yaw from sin/cos survives wrap (350->10 = +20 deg), sign "
          "pinned, pitch and forward distance exact")


def test_depth_mode_separates_rotation_from_translation():
    """O. THE WHOLE ARGUMENT FOR THE FACTORISATION, as a contract.

    Rotation-induced flow carries NO depth information and dominates in first
    person. Translation-induced flow carries all of it. Asked for a raw
    2-channel field, the head must learn that separation from scratch and will
    mostly learn the rotation, because that is where the error mass is.

    So the geometry must make it structural:
      * pure rotation  -> flow INDEPENDENT of depth
      * pure translation -> flow PROPORTIONAL to inverse depth

    If either fails, `flow_mode: depth` is not buying what it claims.
    """
    H = 8
    near = torch.full((1, 1, H, H), 0.9)      # close
    far = torch.full((1, 1, H, H), 0.1)       # distant
    fov = math.radians(70.0)

    rot = torch.tensor([[0.3, 0.0, 0.0]])     # yaw only
    fn = flow_from_depth(near, rot, fov)
    ff = flow_from_depth(far, rot, fov)
    assert float((fn - ff).abs().max()) < 1e-6, (
        "under pure rotation the field must not depend on depth — turning "
        "your head tells you nothing about how far anything is")

    trn = torch.tensor([[0.0, 0.0, 0.5]])     # forward only
    fn = flow_from_depth(near, trn, fov).norm(dim=1).mean()
    ff = flow_from_depth(far, trn, fov).norm(dim=1).mean()
    assert float(fn) > float(ff) * 5.0, (
        f"under pure translation, near must sweep faster than far "
        f"({float(fn):.4f} vs {float(ff):.4f})")

    # And the head in depth mode emits ONE channel, bounded and positive.
    h = FlowHead(8, 3, out_size=H, hidden_dim=32, base_channels=16, mode="depth")
    d = h.inv_depth(torch.randn(2, 8), torch.zeros(2, 3))
    assert d.shape == (2, 1, H, H) and float(d.min()) > 0.0 and float(d.max()) <= 1.0
    # Zero-init => uniform mid-distance prior, and zero ego => exactly zero flow.
    assert abs(float(d.mean()) - 0.5) < 1e-6
    z = h(torch.randn(2, 8), torch.zeros(2, 3), ego=torch.zeros(2, 3))
    assert float(z.abs().max()) == 0.0, "no motion must mean no flow, exactly"
    print(f"  O. rotation depth-independent; translation near/far "
          f"{float(fn) / max(float(ff), 1e-9):.0f}x; depth prior 0.5, "
          f"zero ego -> zero flow")


def test_goal_replay_threshold_selects_log_sized_events():
    """P. THE BATCH THAT ACTUALLY CONTAINS A LOG.

    `_break_reward` already separates perfectly: log 20.0 + 0.5/tick, ore
    10.0, EVERYTHING ELSE exactly 0.0. So the raw env reward is genuinely
    sparse and log-shaped — but the buffer does not store the raw env reward
    for the stream that matters. Both loop bodies store `prim_extrinsic` for
    stream 0, which carries magnet shaping, approach and goal-dwell on top,
    and those are nonzero on most steps. Meanwhile `_find_reward_starts` was
    called with its 1e-3 default.

    MEASURED here: with shaping-sized reward on every step and three log
    breaks in 400, the 1e-3 pool is 396 of 396 possible windows — goal
    replay was EXACTLY uniform sampling, on a run whose entire history holds
    ~35 log breaks.
    """
    from developmental_ai.world_model.replay_buffer import ReplayBuffer
    OD, SEQ = 8, 4
    b = ReplayBuffer(capacity=2000, obs_dim=OD, action_dim=2)
    for i in range(400):
        r = 0.4                                    # shaping-sized, every step
        if i in (100, 200, 300):
            r = 20.0 + 0.5 * 60                    # a felled log
        b.add(np.zeros(OD, np.float32), 0, r, False)

    lo = len(b._find_reward_starts(SEQ, threshold=1e-3))
    hi = len(b._find_reward_starts(SEQ, threshold=5.0))
    assert lo >= (b.size - SEQ) - 1, (
        f"the old default should select essentially everything, got {lo}")
    assert 0 < hi <= 3 * SEQ, f"log-sized threshold selected {hi} windows"
    assert lo > hi * 10, (
        f"threshold must actually discriminate: {lo} vs {hi}")

    batch = b.sample_sequences(8, SEQ, reward_fraction=0.5, reward_threshold=5.0)
    assert batch["reward_pool"] == hi, (
        "the pool size must be reported, or nobody can see this working")

    import yaml
    cfg = yaml.safe_load(open(CFG))
    thr = float(cfg.get("world_model", {}).get("goal_replay_threshold", 1e-3))
    if float(cfg.get("world_model", {}).get("goal_replay_fraction", 0.0)) > 0:
        assert thr >= 5.0, (
            f"goal_replay_threshold is {thr}: below a log break (20.0) the "
            f"stratum re-dilutes to ordinary shaped steps")
    print(f"  P. pool {lo} windows at 1e-3 -> {hi} at 5.0 ({lo // max(1, hi)}x "
          f"dilution removed); live config at {thr}")


def test_every_switch_reverts():
    """Q. Flexibility has to be assertable or it is just a claim.

    Wave 2's defaults must reproduce Wave 1 term for term. Not "close" —
    identical, because the only honest meaning of an opt-in knob is that
    leaving it alone changes nothing.
    """
    IMGL = 16

    def build(seed, **kw):
        torch.manual_seed(seed)
        d = dict(obs_dim=3 * IMGL * IMGL, action_dim=4, stochastic_size=4,
                 stochastic_classes=4, deterministic_size=32, hidden_dim=32,
                 pixel_obs=True, image_size=IMGL, film_conditioning=True,
                 min_grid=4, coord_channels=True, readout_channels=8,
                 flow_head=True, flow_size=8, flow_photo_size=IMGL)
        d.update(kw)
        return WorldModel(**d)

    torch.manual_seed(7)
    B, L = 2, 4
    obs = torch.rand(B, L, 3 * IMGL * IMGL)
    act = F.one_hot(torch.randint(0, 4, (B, L)), 4).float()
    rew, cont = torch.zeros(B, L), torch.ones(B, L)

    wave1 = build(3)
    explicit = build(3, flow_mode="raw", flow_automask=False,
                     flow_scales=[1], flow_strides=[1])
    # RESEED BEFORE EACH FORWARD. compute_loss is STOCHASTIC — the RSSM's
    # straight-through categorical samples — so two sequential calls draw
    # different noise and would differ even for identical models. This test
    # found that the hard way, which is the correct order of events.
    def _loss(m):
        torch.manual_seed(11)
        return m.compute_loss(obs, act, rew, cont)
    a = _loss(wave1)
    c = _loss(explicit)
    for k in ("reconstruction", "flow", "flow_smooth", "total"):
        assert abs(float(a[k]) - float(c[k])) < 1e-6, (
            f"explicit Wave 1 settings changed `{k}`: "
            f"{float(a[k]):.8f} vs {float(c[k]):.8f}")

    # And each switch, turned on, must actually DO something.
    on = build(3, flow_automask=True, flow_scales=[1, 2])
    d = _loss(on)
    assert abs(float(d["flow"]) - float(a["flow"])) > 1e-9, (
        "the Wave 2 switches are configured but inert")
    print(f"  Q. defaults reproduce Wave 1 exactly (flow "
          f"{float(a['flow']):.6f} == {float(c['flow']):.6f}); switches on "
          f"changes it to {float(d['flow']):.6f}")


def test_config_is_coherent():
    """The live config must not enable half of this.

    Three of these knobs change PARAMETER SHAPES. Enabling the flow head
    without the spatial grid, or the body without the flow head, is not an
    error anywhere in the code — it is just a worse model that looks
    configured.
    """
    import yaml
    cfg = yaml.safe_load(open(CFG))
    wm = cfg.get("world_model", {})
    cur = cfg.get("curiosity", {})

    if wm.get("flow_head"):
        assert wm.get("proprio"), (
            "flow_head without world_model.proprio: the model cannot tell "
            "'I turned' from 'the world spun', which is the discrimination "
            "the flow head exists to make")
        assert int(wm.get("spatial_grid", 4)) >= 8, (
            "flow_head at a 4x4 bottleneck: one cell covers a 32x32 px slab "
            "and there is no spatial structure for the field to attach to")
        assert wm.get("coord_channels"), (
            "flow_head without coord_channels leaves all position "
            "information to the final Linear")
        assert float(cur.get("flow_error_weight", 1.0)) >= 0.0
        assert int(wm.get("flow_photo_size", 0)) <= int(
            cfg["environment"]["image_size"]), (
            "flow_photo_size above the frame size would upsample to compare")
    print(f"  H. config coherent: grid={wm.get('spatial_grid')} "
          f"coords={wm.get('coord_channels')} flow={wm.get('flow_head')} "
          f"body={wm.get('proprio')} lambda={wm.get('recon_residual_lambda')} "
          f"lp_weight={cur.get('flow_error_weight')}")


if __name__ == "__main__":
    test_zero_motion_zero_income()
    test_constant_sense_pays_nothing()
    test_sky_aperture_pays_nothing()
    test_reset_splice_pays_nothing()
    test_source_contracts()
    test_position_is_representable()
    test_depth_ordering_is_learned()
    test_surprise_map_localizes()
    test_horizon_term_and_the_floor_that_hid_it()
    test_buffer_proprio_column()
    test_zero_flow_is_the_identity_warp()
    test_automask_masks_the_right_pixels()
    test_multiscale_is_structural()
    test_longer_baselines_are_off_and_why()
    test_ego_from_proprio_is_sign_safe()
    test_depth_mode_separates_rotation_from_translation()
    test_goal_replay_threshold_selects_log_sized_events()
    test_every_switch_reverts()
    test_config_is_coherent()
    print("[perspective] ALL PASS")
