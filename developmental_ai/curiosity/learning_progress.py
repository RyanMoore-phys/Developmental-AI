"""
Learning-Progress curiosity — the Rung 3 alternative to raw-novelty (ICM).

Raw-novelty curiosity rewards prediction ERROR, so an uncontrollable noise source
(the "noisy TV") pays out forever and traps the agent. Learning-progress (LP)
curiosity instead rewards the REDUCTION of prediction error over repeated visits
to a state — i.e. *getting better at predicting*, which is what actually signals
masterable structure (Oudeyer/IMGEP). On pure noise you never get better, so
LP → 0 there; in learnable regions error falls with experience, so LP > 0 until
mastered (then → 0, and the agent moves on).

DESIGN — a controlled swap, not a different model.
This subclasses IntrinsicCuriosityModule and overrides ONLY
`compute_intrinsic_reward`. It reuses the exact same feature encoder + forward
model + `train_step`, so the per-transition prediction error is computed
identically to the novelty arm; the *only* difference between the two Rung-3 arms
is novelty (raw error) vs LP (error reduction). That makes the comparison clean.

LP per transition:
  * bucket the transition by a coarse hash of its observation;
  * keep a short history of this bucket's recent forward-model errors;
  * LP = max(0, mean(older half) − mean(recent half))  — the drop in error.
Crucially, the noisy TV corrupts the observation every step, so each TV
transition lands in a NEW (singleton) bucket → no history → LP = 0. Learnable
states recur → their buckets accumulate → error falls → LP > 0. The asymmetry
(noise = always singleton, learnable = recurs) is what makes LP ignore the TV.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from developmental_ai.curiosity.icm import IntrinsicCuriosityModule


class LearningProgressCuriosity(IntrinsicCuriosityModule):
    """ICM whose intrinsic reward is LEARNING PROGRESS (error reduction per state
    bucket) rather than raw prediction error. Drop-in for IntrinsicCuriosityModule."""

    def __init__(self, *args, lp_history: int = 30, lp_min_samples: int = 4,
                 lp_bucket_round: int = 1, lp_max_buckets: int = 100_000,
                 lp_pixel_pool: int = 4, lp_proto_thresh: float = 0.025,
                 lp_max_protos: int = 512, lp_abs_frac: float = 0.0,
                 action_conditional: bool = False, null_action: int = 0,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # ---- ABSOLUTE PROGRESS FLOOR (2026-08-02) ------------------------
        # LP is a DERIVATIVE, so a world the model has mastered should pay
        # exactly nothing. It did not. The significance gate below is purely
        # RELATIVE (drop > k * the bucket own noise), so on a flat, mastered
        # bucket it still passes a share of pure jitter by construction — and
        # the normalizer then divides by max(running_std, 1e-3) and re-inflates
        # that jitter into a real-looking reward.
        #
        # MEASURED LIVE on a standing-still agent: running_std 1.42e-05, i.e.
        # 70x UNDER the 1e-3 floor, raw prediction error 2.64e-03 — and the
        # base term paid +0.1321/step, 95% of the entire intrinsic drive, for
        # doing nothing. Standing still was the most profitable thing
        # available and no shaping term could outbid it.
        #
        # This requires the error drop to ALSO be a meaningful FRACTION of the
        # bucket own current error. Unit-free and bucket-local, like the
        # existing gate. Real learning drops are large relative to the error;
        # jitter on a mastered bucket is not. 0.0 = off = byte-identical.
        self.lp_abs_frac = float(lp_abs_frac)
        # ---- ACTION-CONDITIONAL CURIOSITY (2026-08-04) -------------------
        # MEASURED: the policy converged deterministically on `noop` (80% of
        # actions, entropy 0.00) while still collecting +0.2951/step. Doing
        # nothing paid almost as well as acting, so a converged policy
        # correctly chose it.
        #
        # WHY it paid: intrinsic reward is forward-model prediction ERROR, and
        # on a live multiplayer server the world changes without the agent —
        # day/night cycles, mobs, other players, foliage. The env asks for
        # frozen time via TimeInitialCondition(allow_passage_of_time=False),
        # but that is a SERVER-side handler and a foreign Paper server ignores
        # it (confirmed: POV frames from one run show both daylight and
        # night). So ambient change generated surprise every step regardless
        # of what the agent did — a standing income for existing.
        #
        # THE GATE, not the signal. Learning progress is still measured on the
        # true forward error; what this scales is how much of that transition
        # the agent's ACTION can account for:
        #     err_a    = ||f(phi(s), a)      - phi(s')||^2   (what happened)
        #     err_null = ||f(phi(s), a_null) - phi(s')||^2   (had I done nothing)
        #     gate     = relu(err_null - err_a) / (err_null + eps)   in [0,1]
        #   ambient change  -> both equally wrong -> gate ~ 0
        #   action caused it-> err_a << err_null  -> gate -> 1
        #   THE AGENT DID NOTHING -> identical inputs -> gate EXACTLY 0
        # That last property is the point: noop can no longer earn anything,
        # by construction rather than by tuning.
        #
        # This gates REWARD, never perception: the world model and the ICM
        # still train on every frame, so learning to predict mobs, weather and
        # other players is unaffected — the agent simply stops being PAID for
        # the world moving on its own.
        self.action_conditional = bool(action_conditional)
        self.null_action = int(null_action)
        self.last_action_attribution = 0.0
        self.lp_history = int(lp_history)
        self.lp_min_samples = int(lp_min_samples)
        self.lp_bucket_round = int(lp_bucket_round)
        self.lp_max_buckets = int(lp_max_buckets)
        # PIXEL bucketing (July 2026, Minecraft curiosity redesign): rounding
        # a 49k-dim float frame makes EVERY frame a singleton bucket, so LP
        # would be permanently zero on pixels; and quantized-hash bucketing is
        # boundary-fragile (a crack-sized overlay flips >=1 of the pooled
        # cells' bins almost surely — measured 0/40 stable). Instead:
        # NEAREST-PROTOTYPE bucketing — a tiny online vector quantizer over
        # block-mean-pooled frames (C*pool*pool dims). A frame joins the
        # nearest prototype if the mean-abs distance is under lp_proto_thresh,
        # else becomes a new prototype (LRU-evicted at lp_max_protos, history
        # dropped with it). Crack overlays move the pooled vector by ~0.01 —
        # DETERMINISTICALLY inside the threshold — so all crack stages of a
        # scene share one bucket, and "getting better at predicting breaking"
        # is exactly the error-drop LP measures there. Distinct scenes differ
        # by >>thresh -> separate buckets. Unlearnable noise either lands in
        # singleton buckets (no history) or merges into one bucket whose
        # error never declines — LP = 0 either way (noisy-TV immunity).
        self.lp_pixel_pool = int(lp_pixel_pool)
        self.lp_proto_thresh = float(lp_proto_thresh)
        self.lp_max_protos = int(lp_max_protos)
        self._protos: dict[int, np.ndarray] = {}   # proto_id -> pooled vec
        self._proto_used: dict[int, int] = {}      # proto_id -> last tick
        self._proto_next_id = 0
        self._proto_tick = 0
        self._bucket_err: dict[int, deque] = {}

    def _pooled_vec(self, obs_row: np.ndarray) -> "np.ndarray | None":
        c = int(getattr(self, "image_channels", 3))
        side = int(getattr(self, "image_size", 64))
        p = self.lp_pixel_pool
        try:
            img = obs_row.reshape(c, side, side)
            k = side // p
            return img[:, : k * p, : k * p].reshape(
                c, p, k, p, k).mean(axis=(2, 4)).flatten()
        except Exception:
            return None  # malformed frame

    def _bucket_key(self, obs_row: np.ndarray, update_state: bool = True) -> int:
        if getattr(self, "pixel_obs", False):
            v = self._pooled_vec(obs_row)
            if v is not None:
                if self._protos:
                    ids = list(self._protos.keys())
                    arr = np.stack([self._protos[i] for i in ids])
                    d = np.abs(arr - v).mean(axis=1)
                    j = int(d.argmin())
                    if d[j] < self.lp_proto_thresh:
                        if update_state:
                            self._proto_tick += 1
                            self._proto_used[ids[j]] = self._proto_tick
                        return ids[j]
                if not update_state:
                    return -1   # read-only miss: no history -> LP 0, no mint
                # new prototype (evict least-recently-used + its history)
                self._proto_tick += 1
                if len(self._protos) >= self.lp_max_protos:
                    old = min(self._proto_used, key=self._proto_used.get)
                    self._protos.pop(old, None)
                    self._proto_used.pop(old, None)
                    self._bucket_err.pop(old, None)
                pid = self._proto_next_id
                self._proto_next_id += 1
                self._protos[pid] = v.astype(np.float32)
                self._proto_used[pid] = self._proto_tick
                return pid
        # VECTOR path: coarse discretization so a recurring state hashes to a
        # stable bucket, while noise-corrupted observations hash uniquely.
        return hash(np.round(obs_row, self.lp_bucket_round).tobytes())

    def _prune_buckets(self):
        # Reclaim memory from the flood of singleton (noise) buckets; keep the
        # informative ones that have accumulated history.
        if len(self._bucket_err) <= self.lp_max_buckets:
            return
        self._bucket_err = {k: v for k, v in self._bucket_err.items() if len(v) > 1}

    def compute_intrinsic_reward(self, obs, action, next_obs,
                                 update_state: bool = True):
        with torch.no_grad():
            features = self.encoder(obs)
            next_features = self.encoder(next_obs)
            predicted_next = self.forward_model(features, action)
            pred_error = F.mse_loss(
                predicted_next, next_features, reduction="none"
            ).mean(dim=-1)

            # RAW forward-model surprise, exposed for consumers that need
            # "how well does the WM predict THIS" rather than the shaped
            # curiosity signal (skill wm_fidelity; the occlusion split).
            # THE BASE CLASS SETS THIS TOO — but this subclass OVERRIDES
            # compute_intrinsic_reward entirely, so the parent's assignment
            # never ran and `last_pred_error` sat at its 0.0 init forever.
            # Consequences, both silent: the curiosity split read
            # `+0.0000 vs +0.0000 (ratio nan)`, and skill wm_fidelity — whose
            # whole purpose is measuring predictability — was fed a constant
            # zero. An overridden method silently orphans every side effect
            # the parent performed.
            self.last_pred_error = float(pred_error.mean().item())
            # counterfactual: what would we have predicted had we done nothing?
            _gate = None
            if self.action_conditional:
                _null = torch.zeros_like(action)
                if _null.dim() == 2 and _null.shape[1] > self.null_action:
                    _null[:, self.null_action] = 1.0      # discrete one-hot
                # continuous: all-zeros IS the null action, nothing to set
                _err_null = F.mse_loss(
                    self.forward_model(features, _null), next_features,
                    reduction="none").mean(dim=-1)
                _gate = torch.clamp(
                    (_err_null - pred_error) / (_err_null + 1e-8),
                    min=0.0, max=1.0)
                self.last_action_attribution = float(_gate.mean().item())
            errs = pred_error.detach().cpu().numpy()
            obs_np = obs.detach().cpu().numpy()
            lp = np.zeros(len(errs), dtype=np.float32)
            for i in range(len(errs)):
                key = self._bucket_key(obs_np[i], update_state=update_state)
                hist = self._bucket_err.get(key)
                if hist is None:
                    if not update_state:
                        continue        # read-only: unseen bucket -> LP 0
                    hist = deque(maxlen=self.lp_history)
                    self._bucket_err[key] = hist
                if len(hist) >= self.lp_min_samples:
                    h = np.fromiter(hist, dtype=np.float32)
                    half = len(h) // 2
                    older = float(h[:half].mean())
                    recent = float(h[half:].mean())
                    # SIGNIFICANCE GATE (audit fix): a mastered bucket's flat
                    # error still jitters, so older-half > recent-half on ~half
                    # of visits by pure chance — max(0, drop) then had strictly
                    # positive expectation forever (farmable fake curiosity on
                    # known scenes). Require the drop to clear a fraction of the
                    # bucket's OWN error spread; real learning drops dwarf the
                    # jitter, chance ticks do not. Unit-free (bucket-local).
                    drop = older - recent
                    noise = float(h.std())
                    # BOTH gates must pass: the drop must clear the bucket
                    # noise (relative significance) AND be a real fraction of
                    # the error still outstanding (absolute progress). The
                    # second is what makes a mastered bucket pay exactly 0
                    # instead of an amplified jitter income.
                    _sig = drop > float(getattr(self, "lp_sig_k", 0.5)) * noise
                    _abs = drop > (float(getattr(self, "lp_abs_frac", 0.0))
                                   * max(recent, 1e-12))
                    if _sig and _abs:
                        lp[i] = drop           # error reduction = progress
                if update_state:
                    hist.append(float(errs[i]))         # counts next time
            if update_state:
                self._prune_buckets()

            reward = torch.clamp(
                torch.as_tensor(lp, dtype=torch.float32, device=obs.device),
                max=self.reward_clip)

            # Same adaptive normalization as the novelty arm, so the two arms feed
            # the reward mixer on a comparable scale (only the SHAPE differs):
            # median-centered, std-scaled, clamped at ZERO from below. LP is
            # mostly zeros, so the running median is ~0 and this reduces to a
            # std-scale — zero-progress transitions (including FIRST VISITS to
            # genuinely new states) get 0, never negative (the mean-centering
            # bug, CODE_AUDIT_2026-07.md §H.3), and there is no all-positive
            # baseline to farm (the 2026-07-12 regression).
            if update_state:
                self._update_stats(reward)
            if self._running_std > 1e-8:
                # std FLOOR (audit fix): the pool is mostly zeros (scout
                # singleton buckets + immature buckets), so an unfloored std
                # sits at noise-tick magnitude and amplifies residual jitter
                # to O(clip). The floor caps that gain; real drops still
                # saturate the clip.
                reward = (reward - self._running_median) / max(
                    self._running_std, 1e-3)
                reward = torch.clamp(reward, 0.0, self.reward_clip)
            # ATTRIBUTION LAST, so it scales the finished curiosity signal
            # rather than distorting the LP statistics that feed the running
            # median/std (those must keep tracking the TRUE forward error, or
            # the normalizer would chase its own gate).
            if _gate is not None:
                reward = reward * _gate
            # Scale AFTER normalization (pre-normalization scaling cancels).
            return reward * self.reward_scale
