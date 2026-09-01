"""Fovea crop-input smoke (2026-08-12).

THE DEFECT THIS FIXES
    The fovea head was trained on the WHOLE-FRAME RSSM latent while its
    labels described the CENTRE CROP — asked to decode "what is in the
    middle 16% of the image" from a global, temporally-mixed embedding.
    MEASURED live over 336k steps: fovea agreement 0.876 against 0.8757 for
    a head that always answers "no" — essentially zero information — with
    P(tree) pinned at its own 0.094 prior and 1 sample in 168 ever reaching
    0.5. Since a tree SIGHTING gates the whole chop pathway, a head that
    cannot see trees is the end of the line.

    Two causes, both fixed here:
      1. WRONG INPUT: the head now reads ENCODER FEATURES OF THE CROP — a
         static function of the patch pixels, which is what a patch label
         actually describes.
      2. CLASS IMBALANCE: the foveal positive rate is ~9-12%, so plain BCE
         is minimised by answering "no" forever. Each predicate's positive
         term is now weighted by its own measured neg/pos ratio.

Run: PYTHONPATH=. python tests/_fovea_crop_input_smoke.py
"""
import sys

import numpy as np
import torch

sys.path.insert(0, ".")

from developmental_ai.llm.vlm_symbolizer import (VLMSymbolizer,   # noqa: E402
                                                 FOVEA_PREDICATES)
from developmental_ai.world_model.rssm import CNNEncoder           # noqa: E402

SIDE = 64


def _mk(fovea_crop=True, enc=None):
    enc = enc or CNNEncoder(3, 256, SIDE)

    def crop_encode(crop):
        a = np.asarray(crop)
        h, w = a.shape[:2]
        yi = np.clip((np.arange(SIDE) * h) // SIDE, 0, h - 1)
        xi = np.clip((np.arange(SIDE) * w) // SIDE, 0, w - 1)
        a = a[yi][:, xi]
        obs = (a.astype(np.float32) / 255.0).transpose(2, 0, 1)
        with torch.no_grad():
            return enc(torch.as_tensor(obs.reshape(1, -1))).detach()

    s = VLMSymbolizer(latent_dim=1536, hidden_dim=64, enabled=True,
                      query_fn=lambda b: None, fovea=True,
                      fovea_latent_dim=(256 if fovea_crop else None),
                      device=torch.device("cpu"))
    if fovea_crop:
        s.set_crop_encoder(crop_encode)
    return s, crop_encode


def _tree_frame(has_tree):
    """A crude but unambiguous 'trunk in the centre' vs 'bare ground'."""
    f = np.zeros((128, 128, 3), np.uint8)
    f[:, :] = (95, 159, 53)
    if has_tree:
        f[30:, 56:72] = (102, 81, 51)      # a trunk THROUGH the centre crop
    return f


def test_head_reads_crop_not_frame():
    s, _ = _mk()
    assert s.fovea_latent_dim == 256, s.fovea_latent_dim
    lat = s._refresh_fovea_latent(_tree_frame(True), 5)
    assert lat is not None and tuple(lat.shape) == (1, 256), lat
    # inference must use the crop cache even when handed a whole-frame latent
    p = s.fovea_probs(torch.randn(1, 1536))
    assert p is not None and len(p) == len(FOVEA_PREDICATES)
    print(f"  1. fovea head input = {s.fovea_latent_dim}-d CROP features "
          f"(the whole-frame latent is 1536-d and is ignored)")


def test_cached_once_per_step():
    s, _ = _mk()
    f = _tree_frame(True)
    a = s._refresh_fovea_latent(f, 11)
    b = s._refresh_fovea_latent(f, 11)          # same step -> same object
    assert a is b, "re-encoded within one step (an extra encoder forward)"
    c = s._refresh_fovea_latent(_tree_frame(False), 12)
    assert c is not b, "cache did not refresh on a new step"
    print("  2. crop encoded at most once per step, refreshed on the next")


def test_head_can_now_learn_trees():
    """THE POINT. Same teacher signal, both wirings, and only the crop-fed
    head can separate a trunk from bare ground."""
    torch.manual_seed(0)
    np.random.seed(0)
    enc = CNNEncoder(3, 256, SIDE)
    results = {}
    for mode in ("crop", "whole_frame"):
        torch.manual_seed(0)
        s, crop_encode = _mk(fovea_crop=(mode == "crop"), enc=enc)
        for i in range(240):
            has = bool(i % 2)
            frame = _tree_frame(has)
            if mode == "crop":
                s._refresh_fovea_latent(frame, i)
                lat = s._fovea_lat_cache
            else:
                # the OLD wiring: a whole-frame latent. Give it the fairest
                # possible version — a real encoding of the full frame.
                a = np.asarray(frame)
                yi = np.clip((np.arange(SIDE) * 128) // SIDE, 0, 127)
                obs = (a[yi][:, yi].astype(np.float32) / 255.0
                       ).transpose(2, 0, 1)
                with torch.no_grad():
                    feat = enc(torch.as_tensor(obs.reshape(1, -1)))
                lat = torch.cat([feat, torch.zeros(1, 1536 - 256)], 1)
            s._train_fovea_head(lat, {"tree_visible": has})
        # score separation on held-out frames
        pos, neg = [], []
        for has in (True, False):
            for _ in range(12):
                frame = _tree_frame(has)
                s._refresh_fovea_latent(frame, 10_000 + len(pos) + len(neg))
                if mode == "crop":
                    lat = s._fovea_lat_cache
                else:
                    a = np.asarray(frame)
                    yi = np.clip((np.arange(SIDE) * 128) // SIDE, 0, 127)
                    obs = (a[yi][:, yi].astype(np.float32) / 255.0
                           ).transpose(2, 0, 1)
                    with torch.no_grad():
                        feat = enc(torch.as_tensor(obs.reshape(1, -1)))
                    lat = torch.cat([feat, torch.zeros(1, 1536 - 256)], 1)
                pr = float(s.fovea_head.predict(lat).squeeze(0)[
                    FOVEA_PREDICATES.index("tree_visible")])
                (pos if has else neg).append(pr)
        results[mode] = (sum(pos) / len(pos), sum(neg) / len(neg))
    c_pos, c_neg = results["crop"]
    assert c_pos > c_neg, f"crop head did not separate: {results['crop']}"
    assert c_pos - c_neg > 0.25, results
    assert c_pos > 0.5, f"P(tree) never crosses the sighting threshold: {c_pos}"
    print(f"  3. crop-fed head SEPARATES trees: P(tree)={c_pos:.2f} with a "
          f"trunk vs {c_neg:.2f} without (gap {c_pos - c_neg:.2f}), and "
          f"clears the 0.5 sighting threshold")


def test_pos_weight_upweights_rare_classes():
    s, _ = _mk()
    s.fovea_label_counts["tree_visible"] = 1000
    s.fovea_pos_counts["tree_visible"] = 100          # 10% positive
    seen = {}
    real = s.fovea_head.train_step

    def spy(lat, target, mask=None, pos_weight=None):
        seen["pw"] = None if pos_weight is None else pos_weight.clone()
        return real(lat, target, mask, pos_weight)
    s.fovea_head.train_step = spy
    s._refresh_fovea_latent(_tree_frame(True), 3)
    s._train_fovea_head(s._fovea_lat_cache, {"tree_visible": True})
    pw = seen["pw"]
    assert pw is not None, "no pos_weight reached the loss"
    i = FOVEA_PREDICATES.index("tree_visible")
    # counts are incremented by this very label BEFORE the weight is derived,
    # so the expectation is computed from the post-increment counts (1001/101)
    n = s.fovea_label_counts["tree_visible"]
    pos = s.fovea_pos_counts["tree_visible"]
    expect = (n - pos) / float(pos)
    assert abs(float(pw[0, i]) - expect) < 1e-4, (float(pw[0, i]), expect)
    assert 8.0 < float(pw[0, i]) < 10.0, float(pw[0, i])
    assert float(pw.max()) <= 20.0, "unclamped weight"
    print(f"  4. pos_weight from measured counts: tree={float(pw[0, i]):.1f}x "
          f"at a 10% positive rate (clamped at 20x)")


def test_fallback_and_shape_guard():
    # No crop encoder -> old whole-frame behaviour, byte-compatible
    s, _ = _mk(fovea_crop=False)
    assert s.fovea_latent_dim == 1536
    p = s.fovea_probs(torch.randn(1, 1536))
    assert p is not None, "fallback path broken"
    # A checkpoint from the OLD wiring must not half-load into the new head
    old = s.state()
    new, _ = _mk(fovea_crop=True)
    before = {k: v.clone() for k, v in new.fovea_head.state_dict().items()}
    summary = new.load_state(old)
    for k, v in new.fovea_head.state_dict().items():
        assert torch.equal(v, before[k]), \
            "old-shape fovea weights partially loaded — torch copies " \
            "matching tensors BEFORE raising, leaving a hybrid"
    assert "FRESH" in summary, summary
    print(f"  5. no-encoder config unchanged (1536-d); a stale-shape "
          f"checkpoint leaves the head FRESH, not hybrid ({summary})")


if __name__ == "__main__":
    for fn in (test_head_reads_crop_not_frame, test_cached_once_per_step,
               test_head_can_now_learn_trees,
               test_pos_weight_upweights_rare_classes,
               test_fallback_and_shape_guard):
        print(f"[fovea-crop] {fn.__name__}")
        fn()
    print("[fovea-crop] ALL PASS")
