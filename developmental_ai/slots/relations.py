"""Pairwise relations between slots — the plurality fix, stated exactly.

WHAT COULD NOT BE SAID BEFORE
    "The nearer tree is left of the farther one and partly behind leaves."

    Booleans cannot say this. Predicates cannot say this. A scene-level
    grounding head emitting `tree_visible=1, object_left=1` has no place to
    put "the nearer one" because it has no notion of one and another. Only a
    RELATION OVER INSTANCES can, which is why slots had to come first.

    This module is the smallest thing that turns K slots into K*(K-1)
    directed relations, and it is the best structure-per-dimension ratio in
    the whole roadmap: five numbers per ordered pair, carrying facts that no
    amount of per-frame predicate vocabulary could encode.

RESEARCH
    Symbolic Relational Deep RL (arXiv 2009.12462) — nodes carry a single
    NON-SEMANTIC feature and no labels, and relational structure emerges
    from the graph rather than from annotation. Continuous Scene
    Representations (arXiv 2203.17251) and Compositional Multi-Object RL
    with Linear Relation Networks (arXiv 2201.13388) are the same family.

NOTHING HERE IS NAMED
    Every field is a difference of two measured quantities. `occludes` is
    computed from depth ordering and mask overlap, not from a label saying
    which object is in front — the geometry says it.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

# Ordered-pair fields, in wire order. Five per pair.
RELATION_FIELDS: Tuple[str, ...] = (
    "d_bearing",     # how far to the left/right of i is j, in [-1, 1]
    "d_depth",       # inverse-depth difference: positive means j is NEARER
    "occludes",      # does i hide part of j (near + overlapping)
    "size_ratio",    # relative apparent size, squashed to [0, 1]
    "co_motion",     # do they move together (one thing) or apart (two)
)


def slot_relations(geom: Dict[str, torch.Tensor],
                   attn: torch.Tensor,
                   alpha: Optional[torch.Tensor] = None) -> torch.Tensor:
    """-> (B, K*(K-1), len(RELATION_FIELDS)), ordered pairs, i != j.

    DIRECTED, not symmetric: "i occludes j" and "j occludes i" are different
    facts and only one of them is usually true. Collapsing to unordered
    pairs would halve the width and lose exactly the asymmetry that makes
    occlusion informative.

    `alpha` (TSA presence) gates every field. A slot whose object is not
    currently present has no relations — leaving them at their stale values
    would assert a spatial arrangement between a thing that is there and a
    thing that is not.
    """
    bearing, depth = geom["bearing"], geom["depth"]
    size = geom["size"]
    b, k = bearing.shape
    dev = bearing.device

    i_idx, j_idx = [], []
    for i in range(k):
        for j in range(k):
            if i != j:
                i_idx.append(i)
                j_idx.append(j)
    ii = torch.tensor(i_idx, device=dev)
    jj = torch.tensor(j_idx, device=dev)

    d_bearing = (bearing[:, jj] - bearing[:, ii]).clamp(-2.0, 2.0) * 0.5
    d_depth = (depth[:, jj] - depth[:, ii]).clamp(-1.0, 1.0)

    # OVERLAP from the attention masks, which is the only evidence available
    # that two slots are competing for the same pixels.
    a = attn / attn.sum(-1, keepdim=True).clamp(min=1e-8)
    overlap = torch.einsum("bkn,bkn->bk", a[:, ii], a[:, jj]) * attn.shape[-1]
    # i OCCLUDES j when they overlap AND i is nearer (larger inverse depth).
    nearer = (depth[:, ii] - depth[:, jj]).clamp(min=0.0)
    occludes = (overlap.clamp(0.0, 1.0) * nearer.clamp(0.0, 1.0))

    size_ratio = (size[:, jj] / (size[:, ii] + size[:, jj] + 1e-8))

    # CO-MOTION is left to the caller to fill from consecutive frames; with
    # a single frame the honest value is 0.5 ("no evidence either way"),
    # never 0, which would claim they move independently.
    co_motion = torch.full_like(d_bearing, 0.5)

    rel = torch.stack([d_bearing, d_depth, occludes, size_ratio, co_motion],
                      dim=-1)

    if alpha is not None:
        gate = (alpha[:, ii] * alpha[:, jj]).unsqueeze(-1)
        neutral = torch.zeros_like(rel)
        neutral[..., 3] = 0.5        # size ratio: equal
        neutral[..., 4] = 0.5        # co-motion: unknown
        rel = gate * rel + (1.0 - gate) * neutral
    return rel


def relation_width(num_slots: int) -> int:
    return num_slots * (num_slots - 1) * len(RELATION_FIELDS)
