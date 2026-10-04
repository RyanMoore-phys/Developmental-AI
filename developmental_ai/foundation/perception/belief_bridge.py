"""Existing perception -> contracts StateBelief (plan Stage 8 item 1).

CONNECTS, DOES NOT REWRITE
    slots/attention.py   SAViSlots.step() dict + SAViSlots.geometry()
    slots/relations.py   slot_relations()
    spatial/memory.py    EgocentricOccupancy, WalkabilityMap
    world_model inverse depth (FlowHead / flow_from_depth, PERSPECTIVE_LEARNING)

WHAT IS CLAIMED
    1. Every belief names its representation and version (major in
       `representation_version`, full semver in payload), carries a
       PER-VARIABLE uncertainty entry, and is SUPPORTED by at least one
       observation in its own scope at or before its seq. A belief from no
       observation is refused.
    2. Slot uncertainty is derived, not invented: bearing/elevation std is
       the spread of the slot's attention over the grid; presence std is the
       Bernoulli std of the TSA alpha; size/identity are UNKNOWN (no
       calibrated measure exists) rather than silently certain.
    3. DEPTH FROM MONOCULAR FLOW IS UP_TO_SCALE, ALWAYS. Flow magnitude is
       fwd * inv_depth: scaling the world and the step by the same s gives
       the same flow, so no amount of flow fixes s. Inverse depth is emitted
       as a geometry Quantity with scale_status "up_to_scale" (exponent -1)
       and a scale_ref naming the reconstruction; there is no argument that
       makes it metric. Ratios of two such depths are metric (geometry).
    4. ZERO IS NOT UNKNOWN. `SAViSlots.geometry(attn, grid, inv_depth=None)`
       returns depth == 0 for every slot — a value claim ("infinitely far").
       The bridge emits depth UNKNOWN when no depth map was given, and drops
       the depth-derived relation fields (d_depth, occludes) instead of
       copying their zeros.
    5. Spatial memory is a FIELD family belief, declared as such. The
       occupancy grid's cell extent has scale_status "unknown": its ranges
       come from 1/sweep-magnitude (up_to_scale) while re-registration uses
       dead-reckoned blocks — a mixture nobody has calibrated.

Shapes are checked at runtime; torch tensors are accepted by duck typing
(this module imports no torch).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..contracts import INAPPLICABLE, ObsRef, Scope, StateBelief, UNKNOWN
from ..geometry import M, RAD, Quantity, Unit
from .errors import PerceptionError
from .families import (DISCRETE_REPRESENTATION, FIELD_REPRESENTATION,
                       OBJECT_REPRESENTATION, FieldFamily, declare_family)
from .identity import TRACKER_REPRESENTATION
from .versioning import RepresentationRegistry

SLOTS_REPRESENTATION = "savi_slots"
DEPTH_REPRESENTATION = "flow_inverse_depth"
OCCUPANCY_REPRESENTATION = "egocentric_occupancy"
WALKABILITY_REPRESENTATION = "walkability"
INV_LENGTH = M ** -1
SLOT_RELATION_FIELDS = ("d_bearing", "d_depth", "occludes", "size_ratio",
                        "co_motion")
_DEPTH_FIELDS = ("d_depth", "occludes")


def default_registry() -> RepresentationRegistry:
    r = RepresentationRegistry()
    for name, desc in (
            (SLOTS_REPRESENTATION, "SAVi slots: bearing/elevation in [-1,1] grid units"),
            (DEPTH_REPRESENTATION, "inverse depth from monocular flow, up_to_scale"),
            (OCCUPANCY_REPRESENTATION, "egocentric top-down occupancy confidence"),
            (WALKABILITY_REPRESENTATION, "per-bearing passability EMA"),
            (TRACKER_REPRESENTATION, "persistent identity tracks"),
            (FIELD_REPRESENTATION, "grid field + 3x3 linear operator"),
            (DISCRETE_REPRESENTATION, "K-state HMM"),
            (OBJECT_REPRESENTATION, "object family = tracker beliefs")):
        r.declare(name, "1.0.0", desc)
    return r


# --------------------------------------------------------------- helpers
def _np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def _support(scope: Scope, seq: int, support: Sequence[ObsRef]) -> Tuple[ObsRef, ...]:
    if not isinstance(scope, Scope):
        raise PerceptionError("scope must be a contracts Scope")
    sup = tuple(support)
    if not sup:
        raise PerceptionError("a belief needs >= 1 supporting observation")
    for s in sup:
        if not isinstance(s, ObsRef) or s.scope != scope or s.seq > seq:
            raise PerceptionError(f"support {s!r} is not an observation at or "
                                  f"before seq {seq} in scope {scope}")
    return sup


def quantity_to_value(q: Quantity) -> Dict[str, Any]:
    """A Quantity in the record value grammar (Quantity itself is not)."""
    if not isinstance(q, Quantity):
        raise PerceptionError(f"expected a Quantity, got {type(q).__name__}")
    return {"value": q.value.copy(), "unit": q.unit.name,
            "unit_dims": tuple(int(d) for d in q.unit.dims),
            "unit_factor": float(q.unit.factor),
            "frame": q.frame if q.frame is not None else INAPPLICABLE,
            "scale_status": q.scale_status,
            "scale_ref": q.scale_ref if q.scale_ref is not None else INAPPLICABLE,
            "scale_exponent": int(q.scale_exponent),
            "std": UNKNOWN if q.std is None else q.std.copy(),
            "provenance": tuple(q.provenance)}


def quantity_from_value(d: Dict[str, Any]) -> Quantity:
    u = Unit(d["unit"], d["unit_dims"], d["unit_factor"], "decoded")
    st = d["scale_status"]
    return Quantity(d["value"], u,
                    frame=None if d["frame"] is INAPPLICABLE else d["frame"],
                    scale_status=st,
                    scale_ref=None if d["scale_ref"] is INAPPLICABLE else d["scale_ref"],
                    scale_exponent=None if st == "metric" else d["scale_exponent"],
                    std=None if d["std"] is UNKNOWN else d["std"],
                    provenance=d["provenance"])


def inverse_depth_quantity(inv_depth, scale_ref: str, std=None,
                           frame: str = "camera") -> Quantity:
    """Monocular-flow inverse depth as an up_to_scale Quantity. There is
    deliberately no `metric` option: only geometry's calibrate(scale,
    evidence) can make it metric, and that requires evidence."""
    if not isinstance(scale_ref, str) or not scale_ref:
        raise PerceptionError("inverse depth needs a scale_ref naming its "
                              "reconstruction")
    v = _np(inv_depth)
    if not np.all(np.isfinite(v)) or np.any(v < 0):
        raise PerceptionError("inverse depth must be finite and >= 0")
    return Quantity(v, INV_LENGTH, frame=frame, scale_status="up_to_scale",
                    scale_ref=scale_ref, std=std,
                    provenance=("monocular_flow",))


def inverse_depth_from_flow(flow, ego, fov_rad: float, scale_ref: str,
                            min_radial: float = 0.1, min_fwd: float = 1e-6):
    """Invert world_model.rssm.flow_from_depth for inverse depth.

    flow (2, H, W) backward field in grid units; ego (3,) = (d_yaw, d_pitch,
    fwd) in the AGENT'S OWN units. Rotation flow is depth-independent
    (PERSPECTIVE_LEARNING contract O) and is subtracted exactly; the radial
    remainder is fwd * inv_depth * r. Per-pixel least squares:
        inv_depth = <residual, (gx, gy)> / (fwd * (gx^2 + gy^2)).
    Because `fwd` is the agent's own step, whose size in world units is not
    known, the result is up_to_scale. Pixels near the focus of expansion
    (radius < min_radial) carry no depth information: they are returned as
    invalid in the mask rather than as a number. No forward motion -> no
    depth at all: raises, because a pure rotation cannot reveal depth.

    -> (Quantity over the valid pixels (n,), valid mask (H, W) bool)
    """
    f = _np(flow)
    e = _np(ego).reshape(-1)
    if f.ndim != 3 or f.shape[0] != 2:
        raise PerceptionError(f"flow must be (2, H, W), got {f.shape}")
    if e.shape != (3,):
        raise PerceptionError(f"ego must be (3,) d_yaw, d_pitch, fwd, got {e.shape}")
    if abs(e[2]) < min_fwd:
        raise PerceptionError("no forward motion: depth is unobservable from "
                              "flow under pure rotation")
    _, h, w = f.shape
    gy, gx = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w),
                         indexing="ij")
    half = max(float(fov_rad) * 0.5, 1e-6)
    rx = f[0] - e[0] / half
    ry = f[1] - e[1] / half
    r2 = gx ** 2 + gy ** 2
    valid = np.sqrt(r2) >= float(min_radial)
    est = (rx * gx + ry * gy)[valid] / (e[2] * r2[valid])
    return inverse_depth_quantity(np.clip(est, 0, None), scale_ref), valid


def _pool(map2d: np.ndarray, grid: int) -> np.ndarray:
    h, w = map2d.shape
    if h % grid or w % grid:
        raise PerceptionError(f"inv_depth {map2d.shape} not divisible by "
                              f"grid {grid}")
    return map2d.reshape(grid, h // grid, grid, w // grid).mean((1, 3))


# ----------------------------------------------------------------- slots
def slot_beliefs(step_out: Dict[str, Any], geom: Dict[str, Any], *,
                 grid: int, scope: Scope, seq: int,
                 support: Sequence[ObsRef], registry: RepresentationRegistry,
                 inv_depth=None, scale_ref: Optional[str] = None,
                 relations=None) -> List[StateBelief]:
    """One StateBelief per batch element from `SAViSlots.step` output and
    `SAViSlots.geometry`. `inv_depth` (B,1,H,W) must be the same map passed
    to geometry(); `relations` the slot_relations() output (optional)."""
    sup = _support(scope, seq, support)
    attn = _np(step_out["attn"])
    alpha = _np(step_out["alpha"])
    ident = _np(step_out["identity"])
    if attn.ndim != 3:
        raise PerceptionError(f"attn must be (B, K, N), got {attn.shape}")
    b, k, n = attn.shape
    if n != grid * grid:
        raise PerceptionError(f"attn has N={n} cells, grid {grid} needs {grid * grid}")
    if alpha.shape != (b, k) or ident.ndim != 3 or ident.shape[:2] != (b, k):
        raise PerceptionError(f"alpha {alpha.shape} / identity {ident.shape} do "
                              f"not match attn (B={b}, K={k})")
    gb = {key: _np(geom[key]) for key in ("bearing", "elevation", "size", "depth")}
    for key, v in gb.items():
        if v.shape != (b, k):
            raise PerceptionError(f"geom[{key!r}] must be (B, K)=({b},{k}), got {v.shape}")
    w = attn / np.clip(attn.sum(-1, keepdims=True), 1e-8, None)
    lin = np.linspace(-1.0, 1.0, grid)
    gy, gx = np.meshgrid(lin, lin, indexing="ij")
    gx, gy = gx.reshape(1, 1, -1), gy.reshape(1, 1, -1)
    bear_std = np.sqrt(np.clip((w * (gx - gb["bearing"][..., None]) ** 2).sum(-1), 0, None))
    elev_std = np.sqrt(np.clip((w * (gy - gb["elevation"][..., None]) ** 2).sum(-1), 0, None))
    if np.abs((w * gx).sum(-1) - gb["bearing"]).max() > 1e-4:
        raise PerceptionError("geom['bearing'] does not match attn: geometry() "
                              "was computed from a different attention map")
    depth_map = None
    if inv_depth is not None:
        if scale_ref is None:
            raise PerceptionError("inv_depth given without a scale_ref")
        dm = _np(inv_depth)
        if dm.ndim != 4 or dm.shape[:2] != (b, 1):
            raise PerceptionError(f"inv_depth must be (B,1,H,W), got {dm.shape}")
        depth_map = np.stack([_pool(dm[i, 0], grid).reshape(-1) for i in range(b)])
        mean = (w * depth_map[:, None, :]).sum(-1)
        if np.abs(mean - gb["depth"]).max() > 1e-4:
            raise PerceptionError("geom['depth'] does not match the inv_depth "
                                  "map given here")
    rel = None
    if relations is not None:
        rel = _np(relations)
        if rel.shape != (b, k * (k - 1), len(SLOT_RELATION_FIELDS)):
            raise PerceptionError(f"relations must be (B, K(K-1), 5), got {rel.shape}")
    major, semver = registry.stamp(SLOTS_REPRESENTATION)
    out = []
    for i in range(b):
        variables: Dict[str, Any] = {
            "bearing": gb["bearing"][i].copy(),
            "elevation": gb["elevation"][i].copy(),
            "size": gb["size"][i].copy(),
            "presence": alpha[i].copy(),
            "identity": ident[i].copy()}
        unc: Dict[str, Any] = {
            "bearing": bear_std[i], "elevation": elev_std[i],
            "size": UNKNOWN,
            "presence": np.sqrt(np.clip(alpha[i] * (1 - alpha[i]), 0, None)),
            "identity": UNKNOWN}
        if depth_map is None:
            variables["inv_depth"] = UNKNOWN
            unc["inv_depth"] = UNKNOWN
        else:
            dstd = np.sqrt(np.clip((w[i] * (depth_map[i][None] - gb["depth"][i][:, None]) ** 2
                                    ).sum(-1), 0, None))
            q = inverse_depth_quantity(gb["depth"][i], scale_ref, std=dstd,
                                       frame="camera")
            variables["inv_depth"] = quantity_to_value(q)
            unc["inv_depth"] = dstd
        fields = SLOT_RELATION_FIELDS
        if rel is not None:
            r = rel[i]
            if depth_map is None:
                keep = [j for j, f in enumerate(SLOT_RELATION_FIELDS)
                        if f not in _DEPTH_FIELDS]
                r = r[:, keep]
                fields = tuple(SLOT_RELATION_FIELDS[j] for j in keep)
            variables["relations"] = r.copy()
            unc["relations"] = UNKNOWN
        out.append(StateBelief(
            belief_id=(f"{SLOTS_REPRESENTATION}:{scope.environment}/"
                       f"{scope.stream}/{scope.episode}:{seq}:b{i}"),
            representation=SLOTS_REPRESENTATION,
            representation_version=major, variables=variables,
            uncertainty=unc, support=sup,
            payload={"representation_semver": semver, "num_slots": k,
                     "grid": grid, "batch_index": i,
                     "relation_fields": tuple(fields),
                     "uncertainty_kind": {
                         "bearing": "attention_spread_std",
                         "elevation": "attention_spread_std",
                         "presence": "bernoulli_std",
                         "inv_depth": "within_mask_std"}}))
    return out


def slot_detections(step_out: Dict[str, Any], geom: Dict[str, Any],
                    batch_index: int = 0, min_presence: float = 0.5):
    """Slots -> tracker detections: positions (n,2)=(bearing, elevation),
    features (n,D)=identity vectors, and the slot indices kept."""
    alpha = _np(step_out["alpha"])[batch_index]
    keep = np.nonzero(alpha >= float(min_presence))[0]
    pos = np.stack([_np(geom["bearing"])[batch_index],
                    _np(geom["elevation"])[batch_index]], -1)[keep]
    feats = _np(step_out["identity"])[batch_index][keep]
    return pos.reshape(-1, 2), feats.reshape(len(keep), -1), keep


# ----------------------------------------------------------------- depth
def depth_belief(inv_depth, *, scale_ref: str, scope: Scope, seq: int,
                 support: Sequence[ObsRef], registry: RepresentationRegistry,
                 std=None) -> StateBelief:
    sup = _support(scope, seq, support)
    q = inverse_depth_quantity(inv_depth, scale_ref, std=std)
    major, semver = registry.stamp(DEPTH_REPRESENTATION)
    return StateBelief(
        belief_id=f"{DEPTH_REPRESENTATION}:{scope.environment}/{scope.stream}/"
                  f"{scope.episode}:{seq}",
        representation=DEPTH_REPRESENTATION, representation_version=major,
        variables={"inv_depth": quantity_to_value(q)},
        uncertainty={"inv_depth": UNKNOWN if std is None else _np(std)},
        support=sup, payload={"representation_semver": semver})


# --------------------------------------------------------------- spatial
_FIELD = FieldFamily()
OCCUPANCY_DECLARATION = declare_family(
    "field", _FIELD,
    reason="EgocentricOccupancy is a top-down grid by construction; no "
           "entity decomposition is claimed")


def occupancy_belief(occ, *, scope: Scope, seq: int,
                     support: Sequence[ObsRef],
                     registry: RepresentationRegistry) -> StateBelief:
    m = _np(occ.map)
    if m.shape != (occ.grid, occ.grid):
        raise PerceptionError(f"occupancy map {m.shape} != grid {occ.grid}")
    cell = Quantity(float(occ.cell), M, frame="body_egocentric",
                    scale_status="unknown",
                    provenance=("cell_blocks config", "ranges from 1/sweep "
                                "magnitude are up_to_scale"))
    return OCCUPANCY_DECLARATION.belief(
        m, registry, scope, seq, support,
        std=np.sqrt(np.clip(m * (1 - m), 0, None)),
        extra={"cell_extent": quantity_to_value(cell)},
        extra_unc={"cell_extent": UNKNOWN},
        representation=OCCUPANCY_REPRESENTATION,
        payload={"convention": "grid[row=z ahead, col=x right], body at centre",
                 "uncertainty_kind": "bernoulli_std_of_confidence"})


def walkability_belief(walk, *, scope: Scope, seq: int,
                       support: Sequence[ObsRef],
                       registry: RepresentationRegistry) -> StateBelief:
    sup = _support(scope, seq, support)
    w = _np(walk.w)
    if w.shape != (walk.bins,):
        raise PerceptionError(f"walkability {w.shape} != bins {walk.bins}")
    bearings = Quantity(np.arange(walk.bins) * (2 * np.pi / walk.bins), RAD,
                        frame="body_egocentric")
    major, semver = registry.stamp(WALKABILITY_REPRESENTATION)
    return StateBelief(
        belief_id=f"{WALKABILITY_REPRESENTATION}:{scope.environment}/"
                  f"{scope.stream}/{scope.episode}:{seq}",
        representation=WALKABILITY_REPRESENTATION,
        representation_version=major,
        variables={"passable": w.copy(),
                   "bin_bearing": quantity_to_value(bearings)},
        uncertainty={"passable": UNKNOWN, "bin_bearing": UNKNOWN},
        support=sup,
        payload={"representation_semver": semver,
                 "note": "passable is an EMA of interaction outcomes, not a "
                         "posterior; its uncertainty is UNKNOWN, not zero"})
