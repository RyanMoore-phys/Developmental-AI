"""SAVi-style object slots over the world model's spatial feature map.

THE PROBLEM THIS EXISTS TO FIX
    The grounding head emits `tree_visible=1, object_left=1`. That
    representation CANNOT EXPRESS "two trees — one at 5 blocks partly behind
    leaves, one at 15". No quantity, no instance, no relation. The vision
    magnet's own docstring concedes it: "no instance disambiguation of two
    trees in one frame". Everything downstream inherits the flattening — the
    knowledge graph is asserted/retracted booleans, skills key on goal slots,
    the magnet targets a category rather than a thing.

    Slots are what make "two trees" representable at all.

THREE DESIGN CHOICES, EACH FROM A PUBLISHED FAILURE

  1. SLOTS DECODE THE FLOW FIELD, NOT PIXELS.
     DINOSAUR (Seitzer et al., ICLR 2023) found that pixel-space
     reconstruction makes slots latch onto COLOUR AND TEXTURE rather than
     objects. Minecraft is maximally textured — a grass field is thousands
     of near-identical high-frequency tiles — so a pixel-target slot model
     here would confidently segment noise. SAVi (Kipf et al., ICLR 2022)
     showed optical flow is the reconstruction target that works, because
     what moves together IS an object. That is also this project's own
     thesis one level down: objectness earned from motion, not declared.

  2. SLOTS ARE CONDITIONED, NOT RANDOMLY INITIALISED.
     SAVi's central result is that a SINGLE POINT on an object in the first
     frame is enough weak conditioning to make instance segmentation emerge,
     and that fully unsupervised initialisation does not scale. We have two
     such points for free and neither is a label: the FOVEA CENTRE (where
     the agent is already looking) and the PEAK OF THE MOVER RESIDUAL (what
     the ego-motion warp could not explain). The first is attention, the
     second is "something moved on its own". Both are already computed.

  3. IDENTITY IS SEPARATE FROM APPEARANCE.
     Dual-State Slot Attention (arXiv 2606.12601) found that encoding both
     in one vector causes slots to SWAP under motion and occlusion. TSA
     (arXiv 2606.13714) adds a learned activation gate so a slot updates
     only while its object is present, letting slots survive long occlusions
     without any visibility supervision. Both matter here more than in a
     benchmark: an agent walking through a forest occludes and reveals
     constantly, and a slot that re-mints on every occlusion cannot support
     "the tree I was walking toward is still there".

WHAT THIS DOES NOT DO
    It does not name anything. A slot is an index with a vector; what any
    slot MEANS is left to the grounding head, which learns it from the VLM
    and from consequences, exactly as it does today for the whole frame.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotAttention(nn.Module):
    """Iterative competitive attention over a feature map (Locatello et al.).

    The mechanism in one line: slots QUERY the features, the attention is
    normalised OVER SLOTS rather than over positions, and so slots compete
    to explain each location. That single normalisation axis is what makes
    the decomposition happen — without it every slot converges on the same
    global average.
    """

    def __init__(self, num_slots: int, dim: int, iters: int = 3,
                 hidden: int = 128, eps: float = 1e-8):
        super().__init__()
        self.num_slots = int(num_slots)
        self.dim = int(dim)
        self.iters = int(iters)
        self.eps = float(eps)
        self.scale = dim ** -0.5

        self.norm_in = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.gru = nn.GRUCell(dim, dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, dim))

    def forward(self, feats: torch.Tensor, slots: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """feats (B, N, D), slots (B, K, D) -> (slots, attn (B, K, N))."""
        feats = self.norm_in(feats)
        k, v = self.to_k(feats), self.to_v(feats)
        attn = None
        for _ in range(self.iters):
            prev = slots
            q = self.to_q(self.norm_slots(slots))
            logits = torch.einsum("bkd,bnd->bkn", q, k) * self.scale
            # NORMALISED OVER SLOTS (dim=1). This is the whole mechanism: it
            # makes slots compete for each location rather than each slot
            # independently pooling the scene.
            attn = logits.softmax(dim=1)
            weights = attn + self.eps
            weights = weights / weights.sum(dim=-1, keepdim=True)
            updates = torch.einsum("bkn,bnd->bkd", weights, v)
            slots = self.gru(updates.reshape(-1, self.dim),
                             prev.reshape(-1, self.dim)).view_as(prev)
            slots = slots + self.ff(self.norm_pre_ff(slots))
        return slots, attn


class SAViSlots(nn.Module):
    """Conditioned, persistent slots that decode the flow field.

    Call `initial(...)` at an episode boundary and `step(...)` every frame.
    The returned dict carries everything downstream needs:

        slots       (B, K, D)  appearance + identity, concatenated
        identity    (B, K, D)  the stable half, for matching across time
        attn        (B, K, N)  which map cells each slot claims
        alpha       (B, K)     TSA activation — is this slot's object present
        pred_flow   (B, 2, H, W) the field the slots collectively explain
        depth       (B, K)     mean inverse depth inside each slot's mask
        bearing     (B, K)     horizontal centroid in [-1, 1]
        elevation   (B, K)     vertical centroid in [-1, 1]
        size        (B, K)     share of the map a slot claims
    """

    def __init__(self, feat_dim: int, num_slots: int = 6, slot_dim: int = 64,
                 iters: int = 3, grid: int = 8, flow_size: int = 32,
                 max_flow: float = 0.5):
        super().__init__()
        self.num_slots = int(num_slots)
        self.slot_dim = int(slot_dim)
        self.grid = int(grid)
        self.flow_size = int(flow_size)
        self.max_flow = float(max_flow)

        self.in_proj = nn.Sequential(
            nn.Conv2d(feat_dim, slot_dim, 1), nn.ELU())
        # Positional encoding on the map, because slot attention is otherwise
        # permutation-invariant over locations and "the left tree" would be
        # indistinguishable from "the right tree" — which is precisely the
        # distinction this whole module exists to make.
        self.register_buffer("pos", self._pos_grid(self.grid, slot_dim),
                             persistent=False)

        self.attn = SlotAttention(num_slots, slot_dim, iters=iters)

        # CONDITIONING: a 2-d hint (a point on an object) -> an initial slot.
        # SAVi's result is that one point per object is enough, and that
        # unconditioned init does not scale to realistic data.
        self.cond = nn.Sequential(
            nn.Linear(2, slot_dim), nn.ELU(), nn.Linear(slot_dim, slot_dim))
        # Unconditioned slots still exist, for the objects nothing pointed
        # at. Learned rather than random so they are at least consistent.
        self.free_slots = nn.Parameter(
            torch.randn(1, num_slots, slot_dim) * 0.02)

        # IDENTITY vs APPEARANCE (Dual-State Slot Attention). One vector for
        # both makes slots swap under motion and occlusion.
        self.to_identity = nn.GRUCell(slot_dim, slot_dim)
        # TSA activation gate: a slot updates only while its object is
        # present, so identity survives an occlusion instead of being
        # overwritten by whatever is now in front of it.
        self.to_alpha = nn.Sequential(
            nn.Linear(slot_dim * 2, slot_dim), nn.ELU(),
            nn.Linear(slot_dim, 1))

        # SPATIAL-BROADCAST DECODER: each slot paints the whole field plus a
        # mask, and the fields are combined by softmax over slots. The mask
        # is what makes a slot own a REGION rather than smearing everywhere.
        self.dec_pos = nn.Parameter(
            torch.randn(1, slot_dim, flow_size, flow_size) * 0.02)
        self.decoder = nn.Sequential(
            nn.Conv2d(slot_dim, 64, 3, padding=1), nn.ELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ELU(),
            nn.Conv2d(64, 3, 3, padding=1))          # 2 flow + 1 mask logit
        # Zero-init the field head for the same reason FlowHead is zero-init:
        # the correct prior before any movement is "nothing moves", and a
        # random field would scramble the first thousand warps.
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    @staticmethod
    def _pos_grid(g: int, dim: int) -> torch.Tensor:
        ys = torch.linspace(-1.0, 1.0, g)
        xs = torch.linspace(-1.0, 1.0, g)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        base = torch.stack([gx, gy, -gx, -gy], dim=0)          # (4, g, g)
        reps = max(1, dim // 4)
        return base.repeat(reps, 1, 1)[:dim].unsqueeze(0) * 0.5

    def initial(self, feat_map: torch.Tensor,
                hints: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Start an episode. `hints` is (B, H, 2) of (x, y) in [-1, 1].

        The loop supplies two hints for free, neither of them a label: the
        FOVEA CENTRE (where the agent is already looking) and the PEAK OF
        THE MOVER RESIDUAL (what the ego-motion warp could not explain).
        """
        b = feat_map.shape[0]
        slots = self.free_slots.expand(b, -1, -1).clone()
        if hints is not None and hints.numel():
            h = min(int(hints.shape[1]), self.num_slots)
            slots[:, :h] = self.cond(hints[:, :h].to(slots.dtype))
        return {"slots": slots, "identity": slots.clone()}

    def step(self, feat_map: torch.Tensor, state: Dict[str, torch.Tensor]
             ) -> Dict[str, torch.Tensor]:
        """One frame. `feat_map` is the encoder's (B, C, grid, grid)."""
        b = feat_map.shape[0]
        x = self.in_proj(feat_map) + self.pos.to(feat_map.dtype)
        feats = x.flatten(2).transpose(1, 2)                   # (B, N, D)

        slots, attn = self.attn(feats, state["slots"])

        # TSA: is this slot's object actually present right now? A slot whose
        # attention is diffuse is explaining nothing in particular.
        conc = (attn * attn).sum(-1) * attn.shape[-1]          # (B, K)
        alpha = torch.sigmoid(self.to_alpha(
            torch.cat([slots, state["identity"]], dim=-1)).squeeze(-1)
            + (conc - 1.0))
        # IDENTITY UPDATES ONLY WHILE PRESENT. This is what carries a slot
        # through an occlusion instead of re-minting it on the far side.
        ident_new = self.to_identity(
            slots.reshape(-1, self.slot_dim),
            state["identity"].reshape(-1, self.slot_dim)).view_as(slots)
        a = alpha.unsqueeze(-1)
        identity = a * ident_new + (1.0 - a) * state["identity"]

        # ---- decode the flow field the slots collectively explain --------
        s = slots.reshape(b * self.num_slots, self.slot_dim, 1, 1)
        canvas = s + self.dec_pos.to(s.dtype)
        out = self.decoder(canvas)
        flow_k = out[:, :2].view(b, self.num_slots, 2,
                                 self.flow_size, self.flow_size)
        mask = out[:, 2:].view(b, self.num_slots, 1,
                               self.flow_size, self.flow_size).softmax(dim=1)
        pred_flow = torch.tanh((flow_k * mask).sum(dim=1)) * self.max_flow

        return {"slots": slots, "identity": identity, "attn": attn,
                "alpha": alpha, "mask": mask, "pred_flow": pred_flow}

    @staticmethod
    def geometry(attn: torch.Tensor, grid: int,
                 inv_depth: Optional[torch.Tensor] = None
                 ) -> Dict[str, torch.Tensor]:
        """Where each slot is, and how far. attn (B, K, N) -> (B, K) each.

        These four are what the relation features are built from, and they
        are the reason slots buy anything: `bearing` and `depth` per INSTANCE
        are exactly what the scene-level predicates could never carry.
        """
        b, k, n = attn.shape
        w = attn / attn.sum(-1, keepdim=True).clamp(min=1e-8)
        ys = torch.linspace(-1.0, 1.0, grid, device=attn.device)
        xs = torch.linspace(-1.0, 1.0, grid, device=attn.device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        bearing = (w * gx.reshape(1, 1, -1)).sum(-1)
        elevation = (w * gy.reshape(1, 1, -1)).sum(-1)
        # Share of total attention: a big near object claims more cells.
        size = attn.sum(-1) / float(n)
        if inv_depth is None:
            depth = torch.zeros_like(bearing)
        else:
            d = F.adaptive_avg_pool2d(inv_depth, grid).reshape(b, 1, -1)
            depth = (w * d).sum(-1)
        return {"bearing": bearing, "elevation": elevation,
                "size": size, "depth": depth}
