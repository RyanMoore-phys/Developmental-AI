"""Unit cases for the world model's new heads and losses.

CONTRACTS vs UNITS. tests/_perspective_smoke.py argues the DESIGN — that
zero flow is the identity warp, that depth ordering is learned from motion,
that every switch reverts. This file checks the PARTS: shapes, bounds,
gradient reachability, and the degenerate inputs that actually occur (a
sequence shorter than a horizon, a batch of one, a stride longer than the
window).

Run: PYTHONPATH=. python tests/unit/test_world_model_unit.py
"""
import sys

sys.path.insert(0, ".")

import numpy as np
import torch
import torch.nn.functional as F

from tests.unit._runner import case, run_all, close, between, finite, raises
from developmental_ai.world_model.rssm import (
    WorldModel, FlowHead, SensorImageEncoder, CNNEncoder, CNNDecoder,
    flow_warp, photometric, charbonnier, edge_aware_smoothness,
    ego_from_proprio, flow_from_depth, _cnn_channel_ladder, CHARBONNIER_EPS)

IMG = 16
A = 4
LAYOUT = {"pitch": 9, "moved": 10, "head_sin": 11, "head_cos": 12}


def _wm(**kw):
    d = dict(obs_dim=3 * IMG * IMG, action_dim=A, stochastic_size=4,
             stochastic_classes=4, deterministic_size=16, hidden_dim=16,
             pixel_obs=True, image_size=IMG, film_conditioning=True,
             flow_head=True, flow_size=8, flow_photo_size=IMG)
    d.update(kw)
    return WorldModel(**d)


def _seq(wm, b=2, l=6, pd=0):
    obs = torch.rand(b, l, 3 * IMG * IMG)
    act = F.one_hot(torch.randint(0, A, (b, l)), A).float()
    pp = torch.rand(b, l, pd) if pd else None
    return obs, act, torch.zeros(b, l), torch.ones(b, l), pp


# ------------------------------------------------------------- primitives

@case
def channel_ladder_shapes():
    assert _cnn_channel_ladder(64) == [32, 64, 128, 256]
    assert _cnn_channel_ladder(128) == [32, 64, 128, 256, 256]
    assert _cnn_channel_ladder(128, min_grid=8) == [32, 64, 128, 256]
    raises(lambda: _cnn_channel_ladder(100), ValueError, "non power of two")
    raises(lambda: _cnn_channel_ladder(16, min_grid=32), ValueError,
           "min_grid larger than the image")


@case
def encoder_declares_its_own_widths():
    e = CNNEncoder(3, 32, 32, min_grid=8, coord_channels=True,
                   readout_channels=16)
    v, m = e.encode_with_map(torch.zeros(2, 3 * 32 * 32))
    assert v.shape == (2, 32) and e.vec_dim == 32
    assert m.shape[2] == m.shape[3] == e.grid == 8
    assert m.shape[1] == e.map_channels


@case
def decoder_round_trips_every_min_grid():
    for mg in (4, 8):
        d = CNNDecoder(12, 3, 16, 32, min_grid=mg)
        assert d(torch.zeros(1, 12)).shape == (1, 3 * 32 * 32), mg


@case
def charbonnier_floor_is_what_the_code_subtracts():
    z = float(charbonnier(torch.zeros(1)).item())
    close(z, CHARBONNIER_EPS, 1e-9, "charbonnier(0)")
    assert z > 0.0, (
        "if this were 0 the floor subtraction would be dead code and a still "
        "cell would read 1.0 surprise again")


@case
def warp_identity_and_shift():
    img = torch.rand(2, 3, 16, 16)
    close(float((flow_warp(img, torch.zeros(2, 2, 16, 16)) - img).abs().max()),
          0.0, 1e-5, "identity warp")
    # A positive x-flow samples from the RIGHT, so content moves LEFT.
    f = torch.zeros(2, 2, 16, 16)
    f[:, 0] = 2.0 / 15.0                           # exactly one pixel
    w = flow_warp(img, f)
    close(float((w[:, :, :, :-1] - img[:, :, :, 1:]).abs().max()), 0.0, 1e-4,
          "one-pixel shift")


@case
def photometric_returns_two_maps_of_the_right_shape():
    a, b = torch.rand(3, 3, 8, 8), torch.rand(3, 3, 8, 8)
    we, ie = photometric(a, b, torch.zeros(3, 2, 8, 8))
    assert we.shape == ie.shape == (3, 1, 8, 8)
    close(float((we - ie).abs().max()), 0.0, 1e-6,
          "at zero flow the warp error IS the identity error")


@case
def smoothness_is_zero_on_a_constant_field():
    img = torch.rand(2, 3, 8, 8)
    close(float(edge_aware_smoothness(torch.zeros(2, 2, 8, 8), img)), 0.0,
          1e-8, "constant flow")
    assert float(edge_aware_smoothness(torch.randn(2, 2, 8, 8), img)) > 0.0


# --------------------------------------------------------------- flow head

@case
def flow_head_starts_at_exactly_zero():
    for mode, ego in (("raw", None), ("depth", torch.zeros(2, 3))):
        h = FlowHead(8, A, out_size=8, hidden_dim=16, base_channels=16,
                     mode=mode)
        f = h(torch.randn(2, 8), torch.zeros(2, A), ego=ego)
        close(float(f.abs().max()), 0.0, 1e-9, f"{mode} zero-init")


@case
def flow_head_output_channels_follow_mode():
    assert FlowHead(8, A, out_size=8, mode="raw").out.out_channels == 2
    assert FlowHead(8, A, out_size=8, mode="depth").out.out_channels == 1
    raises(lambda: FlowHead(8, A, out_size=8, mode="nope"), ValueError)
    raises(lambda: FlowHead(8, A, out_size=6), ValueError, "non power of two")


@case
def depth_mode_requires_ego():
    h = FlowHead(8, A, out_size=8, mode="depth")
    raises(lambda: h(torch.randn(1, 8), torch.zeros(1, A)), ValueError,
           "depth mode without ego")


@case
def inv_depth_is_bounded_and_starts_at_the_midpoint():
    h = FlowHead(8, A, out_size=8, mode="depth")
    d = h.inv_depth(torch.randn(4, 8), torch.zeros(4, A))
    assert d.shape == (4, 1, 8, 8)
    between(float(d.min()), 0.0, 1.0); between(float(d.max()), 0.0, 1.0)
    close(float(d.mean()), 0.5, 1e-6, "uniform mid-distance prior")


@case
def range_scale_widens_the_bound():
    h = FlowHead(8, A, out_size=8, hidden_dim=16, base_channels=16)
    torch.nn.init.normal_(h.out.weight, std=2.0)
    torch.nn.init.normal_(h.out.bias, std=2.0)
    lat, act = torch.randn(1, 8), torch.zeros(1, A)
    m1 = float(h(lat, act, range_scale=1.0).abs().max())
    m4 = float(h(lat, act, range_scale=4.0).abs().max())
    assert m4 > m1 * 3.0, f"{m1:.3f} -> {m4:.3f}"


# ----------------------------------------------------------------- geometry

@case
def ego_from_proprio_fields():
    def row(yaw, pitch=0.5, moved=0.0):
        v = torch.zeros(1, 13)
        v[0, LAYOUT["pitch"]] = pitch
        v[0, LAYOUT["moved"]] = moved
        v[0, LAYOUT["head_sin"]] = np.sin(yaw)
        v[0, LAYOUT["head_cos"]] = np.cos(yaw)
        return v
    e = ego_from_proprio(row(0.0), row(np.pi / 2), LAYOUT)
    close(float(e[0, 0]), np.pi / 2, 1e-5, "quarter turn")
    e = ego_from_proprio(row(np.radians(350)), row(np.radians(10)), LAYOUT)
    close(float(e[0, 0]), np.radians(20), 1e-5, "wrap-around")
    raises(lambda: ego_from_proprio(row(0.0), row(0.0), {"pitch": 0}),
           ValueError, "incomplete layout")


@case
def flow_from_depth_is_linear_in_translation():
    d = torch.full((1, 1, 8, 8), 0.5)
    f1 = flow_from_depth(d, torch.tensor([[0.0, 0.0, 0.5]]), 1.22)
    f2 = flow_from_depth(d, torch.tensor([[0.0, 0.0, 1.0]]), 1.22)
    close(float((f2 - 2.0 * f1).abs().max()), 0.0, 1e-5,
          "doubling forward motion must double the field")


@case
def flow_from_depth_rotation_ignores_depth_entirely():
    ego = torch.tensor([[0.4, -0.2, 0.0]])
    a = flow_from_depth(torch.full((1, 1, 8, 8), 0.1), ego, 1.22)
    b = flow_from_depth(torch.full((1, 1, 8, 8), 0.9), ego, 1.22)
    close(float((a - b).abs().max()), 0.0, 1e-7)


@case
def zero_ego_gives_zero_flow_at_any_depth():
    for dv in (0.01, 0.5, 0.99):
        f = flow_from_depth(torch.full((1, 1, 8, 8), dv),
                            torch.zeros(1, 3), 1.22)
        close(float(f.abs().max()), 0.0, 1e-9, f"depth {dv}")


# ------------------------------------------------------------- integration

@case
def sensor_image_encoder_compresses():
    e = SensorImageEncoder(3, 32, out_dim=16)
    assert e(torch.rand(2, 3, 32, 32)).shape == (2, 16)


@case
def embed_widths_add_up_with_and_without_images():
    plain = _wm(proprio_dim=7)
    assert plain.sensor_embed_dim == 7
    assert plain.embed(torch.rand(2, 3 * IMG * IMG),
                       torch.rand(2, 7)).shape[1] == plain.rssm.obs_dim
    withimg = _wm(proprio_dim=7 + 3 * 8 * 8,
                  sensor_image_specs=[(7, 3, 8, 8)], sensor_feature_dim=5)
    assert withimg.sensor_embed_dim == 7 + 5
    assert withimg.embed(torch.rand(2, 3 * IMG * IMG),
                         torch.rand(2, 7 + 192)).shape[1] == withimg.rssm.obs_dim


@case
def embed_tolerates_missing_sensors():
    wm = _wm(proprio_dim=7)
    out = wm.embed(torch.rand(2, 3 * IMG * IMG), None)
    assert out.shape[1] == wm.rssm.obs_dim


@case
def horizons_are_sanitised():
    assert _wm(latent_horizons=[1, 0, -5, 2.0]).latent_horizons == [2]
    assert _wm(latent_horizons=None).latent_horizons == []
    assert _wm(latent_horizons=[16, 4, 4]).latent_horizons == [4, 16]


@case
def horizon_is_inert_when_it_cannot_fit():
    wm = _wm(latent_horizons=[32])
    o, a, r, c, _ = _seq(wm, l=6)
    close(float(wm.compute_loss(o, a, r, c)["horizon"]), 0.0, 1e-12)


@case
def prior_divergence_is_zero_for_identical_logits_and_positive_otherwise():
    wm = _wm()
    x = torch.randn(4, wm.rssm.stoch_dim)
    close(float(wm.rssm.prior_divergence(x, x)), 0.0, 1e-5)
    assert float(wm.rssm.prior_divergence(x, torch.randn_like(x))) > 0.0
    # ... and compute_kl_loss has a floor, which is why it is not used there.
    assert float(wm.rssm.compute_kl_loss(x, x)) > 0.1


@case
def every_loss_key_is_finite_and_scalar():
    wm = _wm(latent_horizons=[2], flow_automask=True, flow_scales=[1, 2],
             recon_residual_lambda=2.0)
    o, a, r, c, _ = _seq(wm, l=6)
    losses = wm.compute_loss(o, a, r, c)
    for k, v in losses.items():
        assert v.dim() == 0, f"{k} is not a scalar"
        finite(v.detach().numpy(), k)


@case
def flow_loss_reaches_the_flow_head():
    wm = _wm()
    o, a, r, c, _ = _seq(wm, l=4)
    wm.zero_grad()
    wm.compute_loss(o, a, r, c)["flow"].backward()
    g = sum(float(p.grad.abs().sum()) for p in wm.flow_head.parameters()
            if p.grad is not None)
    assert g > 0.0, "the flow loss does not reach the flow head"


@case
def batch_of_one_and_short_sequences_do_not_crash():
    wm = _wm(latent_horizons=[2])
    for b, l in ((1, 2), (1, 3), (3, 2)):
        o, a, r, c, _ = _seq(wm, b=b, l=l)
        finite(wm.compute_loss(o, a, r, c)["total"].detach().numpy(),
               f"total at b={b} l={l}")


@case
def residual_map_shape_and_gate():
    wm = _wm()
    lat = torch.randn(2, wm.rssm.latent_dim)
    act = F.one_hot(torch.tensor([1, 1]), A).float()
    frame = torch.rand(2, 3 * IMG * IMG)
    r, m = wm.flow_residual(frame, frame.clone(), lat, act, return_map=True,
                            map_size=4)
    assert m.shape == (2, 1, 4, 4)
    close(float(m.abs().max()), 0.0, 1e-9, "static frames must map to zero")
    close(float(r.abs().max()), 0.0, 1e-9, "static frames must score zero")


@case
def probe_map_is_bounded_and_sized():
    for mode in ("raw", "depth"):
        wm = _wm(flow_mode=mode)
        pm = wm.probe_map(torch.randn(2, wm.rssm.latent_dim),
                          F.one_hot(torch.tensor([1, 1]), A).float(),
                          map_size=4)
        assert pm.shape == (2, 1, 4, 4), mode
        between(float(pm.min()), 0.0, 1.0, mode)
        between(float(pm.max()), 0.0, 1.0, mode)


@case
def probe_map_is_none_without_a_flow_head():
    wm = _wm(flow_head=False)
    assert wm.probe_map(torch.randn(1, wm.rssm.latent_dim),
                        torch.zeros(1, A)) is None


@case
def slots_refuse_to_build_without_a_flow_head():
    assert _wm(flow_head=False, slots=True).slots_enabled is False
    assert _wm(flow_head=True, slots=True).slots_enabled is True


if __name__ == "__main__":
    sys.exit(run_all("world-model-unit"))
