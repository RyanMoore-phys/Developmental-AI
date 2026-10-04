"""Differentiable (PyTorch) versions of the quaternion and rotor operations.

Same conventions as transforms.py and rotor.py, verified against them in
tests/_foundation_geometry_smoke.py and with torch.autograd.gradcheck in
float64. Import this module explicitly; the NumPy core never imports torch.

`quat_normalize_t` rejects (raises) a quaternion with norm below `eps`
rather than clamping: a clamp would give a finite but meaningless gradient
through a non-rotation.
"""

from __future__ import annotations

import torch

from .rotor import N_BLADES, PRODUCT_TABLE, _REV_SIGN
from .errors import DegenerateError

_REV_T = torch.tensor(_REV_SIGN.tolist(), dtype=torch.float64)


def quat_normalize_t(q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    if q.shape[-1] != 4:
        raise DegenerateError(f"quaternion must have trailing dim 4, got {tuple(q.shape)}")
    n = torch.linalg.norm(q, dim=-1, keepdim=True)
    if bool((n < eps).any()):
        raise DegenerateError("zero-norm quaternion")
    return q / n


def quat_mul_t(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def quat_rotate_t(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """q v q* for unit q (normalise first if q is a free parameter)."""
    if q.shape[-1] != 4 or v.shape[-1] != 3:
        raise DegenerateError("expected q (...,4) and v (...,3)")
    w = q[..., :1]
    u, v = torch.broadcast_tensors(q[..., 1:], v)
    t = 2.0 * torch.cross(u, v, dim=-1)
    return v + w * t + torch.cross(u, t, dim=-1)


def gp_t(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cl(3,0) geometric product (trailing dim 8)."""
    shape = torch.broadcast_shapes(a.shape, b.shape)
    a = a.expand(shape)
    b = b.expand(shape)
    cols = [None] * N_BLADES
    for i, j, k, s in PRODUCT_TABLE:
        term = s * a[..., i] * b[..., j]
        cols[k] = term if cols[k] is None else cols[k] + term
    return torch.stack(cols, dim=-1)


def rotor_from_quat_t(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    zero = torch.zeros_like(w)
    # blades: 1 e1 e2 e12 e3 e13 e23 e123  ->  w - x e23 + y e13 - z e12
    return torch.stack([w, zero, zero, -z, zero, y, -x, zero], dim=-1)


def rotor_sandwich_t(R: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """R v ~R for rotor(s) R (trailing 8) and vectors v (trailing 3)."""
    z = torch.zeros_like(v[..., 0])
    mv = torch.stack([z, v[..., 0], v[..., 1], z, v[..., 2], z, z, z], dim=-1)
    rev = R * _REV_T.to(dtype=R.dtype, device=R.device)
    out = gp_t(gp_t(R, mv), rev)
    return torch.stack([out[..., 1], out[..., 2], out[..., 4]], dim=-1)
