"""Spatial memory: what was around me, where I am likely to go, what I can
walk through. Roadmap items G2, P2 and P3.

WHY THIS EXISTS
    Every representation in this project has been PER-FRAME. The grounding
    head answers "is a tree visible NOW"; the magnet targets what is on
    screen NOW; the flow senses describe THIS step. Nothing carried the fact
    that a wall was on the left three steps ago, and nothing could — which is
    why "a tree at the edge of frame and a tree dead-centre pay exactly the
    same" was a fair description of the whole stack and not just of one
    reward term.

    These three maps are the first representations in the system that outlive
    the frame they came from.

THE INPUT IS THE AGENT'S OWN ESTIMATE, NOT THE ENGINE'S TRUTH
    The occupancy map is built from the world model's forward-probe magnitude
    (near sweeps fast, far sweeps slow) and re-registered by the agent's own
    sensed ego-motion. Both are things the body produces. Nothing here reads
    a block name, a voxel grid or a true coordinate — the engine's position
    exists in this codebase only as the RED oracle, and only to MEASURE these.

    That means the maps inherit the errors of their inputs, which is correct:
    a mistaken sense of how far away the wall was should produce a mistaken
    map, and it is the oracle's job to say how mistaken.

RESEARCH
    EgoMap (arXiv 2002.02286) establishes the construction — project
    per-frame features into a top-down map and update it through a
    differentiable transform driven by ego-motion — and finds it helps
    exactly on multi-step objectives in 3D. Egocentric Spatial Memory (arXiv
    1807.11929) and Spatially-Enhanced Recurrent Memory (arXiv 2506.05997)
    are the same family. The successor map below is the simplest form of the
    successor representation (J. Neurosci. 38(33):7193): represent a state
    "not by its immediate sensory features, but by the states it predicts".
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


def rotate_translate(grid: np.ndarray, d_yaw: float,
                      dx_cells: float, dz_cells: float) -> np.ndarray:
    """Re-register an egocentric grid after the body moved.

    CONVENTION, stated because half of it was wrong first time round:
    grid[row, col] = grid[z, x]; the body sits at the centre; AHEAD IS
    INCREASING ROW and the body's RIGHT is increasing column. `output[p]` is
    sampled from `input[p + d]`, which is the backward-mapping convention —
    so to bring content one cell CLOSER the offset is POSITIVE.

    THE SIGNS, DERIVED RATHER THAN ARGUED (a numeric probe, not intuition —
    the intuitive phrasing "the world slides backwards" produced exactly the
    wrong translation sign and a map that smeared AWAY from the direction of
    travel):

        advance by f    dz = +f   (a mark 2 ahead moves to 1 ahead)
        strafe right l  dx = +l
        turn right by y angle = -y  (a mark ahead ends up on the LEFT)

    Rotation is the inverse of the body's turn; translation is NOT the
    inverse, because the backward map already carries that inversion. Both
    are pinned by trajectory in tests/_spatial_memory_smoke.py, because this
    is precisely the class of error that looks plausible on a static frame
    and is wrong the moment anything moves.

    Bilinear, and deliberately not a library call: the only dependency in
    this file is numpy, which keeps the whole thing runnable in a test with
    no torch, no Minecraft and no GPU.
    """
    g = grid.shape[0]
    c = (g - 1) / 2.0
    yy, xx = np.meshgrid(np.arange(g, dtype=np.float32),
                         np.arange(g, dtype=np.float32), indexing="ij")
    # Coordinates relative to the body, which sits at the centre.
    rx, rz = xx - c, yy - c
    # Undo the translation first (the world slid by -d), then the rotation.
    rx = rx + dx_cells
    rz = rz + dz_cells
    ca, sa = math.cos(d_yaw), math.sin(d_yaw)
    sx = ca * rx - sa * rz + c
    sz = sa * rx + ca * rz + c

    x0 = np.floor(sx).astype(np.int32)
    z0 = np.floor(sz).astype(np.int32)
    fx = (sx - x0).astype(np.float32)
    fz = (sz - z0).astype(np.float32)
    out = np.zeros_like(grid)
    for dz in (0, 1):
        for dx in (0, 1):
            xi, zi = x0 + dx, z0 + dz
            ok = (xi >= 0) & (xi < g) & (zi >= 0) & (zi < g)
            w = ((fx if dx else 1.0 - fx) * (fz if dz else 1.0 - fz))
            out[ok] += (grid[np.clip(zi, 0, g - 1), np.clip(xi, 0, g - 1)]
                        * w)[ok]
    return out


class EgocentricOccupancy:
    """A rolling top-down map of what was solid around the body. (G2)

    The body sits at the centre facing +row. Cells hold a confidence in
    [0, 1] that something was there, decaying with age — a MEMORY, not a
    survey, so a cell the agent has not seen for a while fades rather than
    asserting emptiness it cannot vouch for.
    """

    def __init__(self, grid: int = 16, cell_blocks: float = 1.0,
                 decay: float = 0.93, fov_rad: float = 1.2217304764,
                 max_range: float = 12.0):
        self.grid = int(grid)
        self.cell = float(cell_blocks)
        self.decay = float(decay)
        self.fov = float(fov_rad)
        self.max_range = float(max_range)
        self.reset()

    def reset(self) -> None:
        self.map = np.zeros((self.grid, self.grid), dtype=np.float32)

    def _scan_to_cells(self, scan: np.ndarray) -> np.ndarray:
        """A per-column sweep magnitude -> a top-down write mask.

        Each image column is a BEARING; the largest sweep magnitude in that
        column is the nearest thing along it, because near things sweep
        fastest. Inverse: range ~ 1/magnitude, which is the same relation the
        depth factorisation makes explicit and the raw head has to learn.
        """
        out = np.zeros((self.grid, self.grid), dtype=np.float32)
        n = scan.shape[0]
        if n == 0:
            return out
        c = (self.grid - 1) / 2.0
        for i in range(n):
            m = float(scan[i])
            if m <= 1e-3:
                continue                      # nothing resolvable this bearing
            rng = min(self.max_range, 1.0 / max(m, 1e-3))
            # Column index -> bearing across the field of view. Column 0 is
            # the LEFT edge, which is -fov/2 from centre.
            ang = (i / max(1, n - 1) - 0.5) * self.fov
            zc = c + (rng * math.cos(ang)) / self.cell
            xc = c + (rng * math.sin(ang)) / self.cell
            zi, xi = int(round(zc)), int(round(xc))
            if 0 <= zi < self.grid and 0 <= xi < self.grid:
                out[zi, xi] = max(out[zi, xi], min(1.0, m * 4.0))
        return out

    def step(self, probe_map: Optional[np.ndarray], d_yaw: float,
             fwd: float, lat: float) -> np.ndarray:
        """Advance one step. `probe_map` is (S, S) sweep magnitudes."""
        # RE-REGISTER FIRST. The body has already moved by the time this
        # frame was taken, so the old map must be brought into the new frame
        # before anything is written into it. Writing first would place this
        # frame's reading at the body's PREVIOUS pose.
        self.map = rotate_translate(
            self.map, -float(d_yaw),
            float(lat) / self.cell, float(fwd) / self.cell)
        # ---- CLAMP AFTER RE-REGISTRATION (found by unit test, 2026-09-19) -
        # Bilinear BACKWARD mapping does not exactly conserve mass: the
        # weight a source cell contributes across all the output cells that
        # read it sums to ~1 on average but can exceed 1 locally, so a
        # rotation can AMPLIFY a cell slightly. Measured on a single lit cell
        # at 0.7 rad, the total grew.
        #
        # That is harmless once and compounding forever: the agent turns
        # constantly, and an amplification above 1/decay would make
        # confidence grow without bound until every cell read "solid". A
        # confidence is a probability-like quantity and belongs in [0, 1]
        # whatever the resampler does, so it is clamped rather than trusted.
        self.map = np.clip(self.map, 0.0, 1.0)
        self.map *= self.decay
        if probe_map is not None and probe_map.size:
            a = np.asarray(probe_map, dtype=np.float32)
            if a.ndim == 2:
                # Nearest obstacle per COLUMN: a 1-D range scan, which is all
                # a top-down map can use from a 2-D image anyway.
                self.map = np.maximum(self.map, self._scan_to_cells(a.max(axis=0)))
        return self.map


class SuccessorMap:
    """Discounted future occupancy of DEAD-RECKONED cells. (P2)

    WHY NOT OVER THE EGOCENTRIC GRID, which would have been the obvious
    place to put it: the egocentric frame re-centres on the body every step,
    so "the cell I am in" is always the middle one and the successor of every
    state is the same state. The SR needs a state space that holds still
    while the agent moves through it. The dead-reckoned frame is exactly that
    — stable within an episode, and built from the body's own sense rather
    than from the engine's coordinates.

    Simplest possible TD form: M(s) <- M(s) + lr * (onehot(s) + gamma*M(s')
    - M(s)), one row at a time, which is all that is needed to answer "where
    am I likely to be soon".
    """

    def __init__(self, grid: int = 8, cell_blocks: float = 8.0,
                 gamma: float = 0.95, lr: float = 0.1):
        self.grid = int(grid)
        self.cell = float(cell_blocks)
        self.gamma = float(gamma)
        self.lr = float(lr)
        self.reset()

    def reset(self) -> None:
        self.m = np.zeros((self.grid * self.grid, self.grid * self.grid),
                          dtype=np.float32)
        self._prev = None

    def _index(self, x: float, z: float) -> int:
        """Dead-reckoned position -> a cell index, WRAPPED.

        Wrapping rather than clipping: the frame has no bounds (the agent can
        walk forever) and clipping would pile every distant position into the
        edge cells, making "far away north" and "far away east" the same
        state. Wrapping aliases distant places instead, which is the same
        trade a periodic positional code makes and is honest about being a
        local map.
        """
        gx = int(math.floor(x / self.cell)) % self.grid
        gz = int(math.floor(z / self.cell)) % self.grid
        return gz * self.grid + gx

    def step(self, dr_x: float, dr_z: float) -> np.ndarray:
        s = self._index(dr_x, dr_z)
        if self._prev is not None:
            p = self._prev
            onehot = np.zeros(self.m.shape[1], dtype=np.float32)
            onehot[p] = 1.0
            target = onehot + self.gamma * self.m[s]
            self.m[p] += self.lr * (target - self.m[p])
        self._prev = s
        return self.m[s]


class WalkabilityMap:
    """Can I go that way? Learned from what happened when I tried. (P3)

    ANNOTATION BY INTERACTION, which is the only kind of label this project
    allows: if the agent commanded forward motion and the body moved, the
    direction it was facing was passable; if it commanded forward and did not
    move, it was blocked. Nobody labels a wall — the wall labels itself by
    stopping you, exactly as _reach_sense is grounded by real break events
    rather than by a rangefinder.

    Research: Learning Affordances from Interactive Exploration (arXiv
    2501.06047) — a successful or failed interaction IS the annotation.

    Indexed by RELATIVE BEARING in the egocentric frame, so the map answers
    "is it clear ahead / to my left / behind", which is what a body needs,
    and needs no world frame at all.
    """

    def __init__(self, bins: int = 8, lr: float = 0.1, prior: float = 0.5):
        self.bins = int(bins)
        self.lr = float(lr)
        self.prior = float(prior)
        self.reset()

    def reset(self) -> None:
        # PRIOR 0.5 = "no opinion". Zero would be a claim that the world is
        # a solid block, which is the belief that would stop an agent from
        # ever trying — and an untried direction is precisely the one worth
        # trying.
        self.w = np.full(self.bins, self.prior, dtype=np.float32)

    def step(self, commanded_move: bool, moved_blocks: float,
             d_yaw: float, move_thresh: float = 0.05) -> np.ndarray:
        """Rotate with the body, then record the outcome dead ahead."""
        if abs(d_yaw) > 1e-6:
            # The bearings are body-relative, so a turn slides the whole map.
            shift = -d_yaw / (2.0 * math.pi) * self.bins
            idx = (np.arange(self.bins) + shift) % self.bins
            lo = np.floor(idx).astype(int) % self.bins
            hi = (lo + 1) % self.bins
            f = (idx - np.floor(idx)).astype(np.float32)
            self.w = self.w[lo] * (1.0 - f) + self.w[hi] * f
        if commanded_move:
            outcome = 1.0 if float(moved_blocks) > move_thresh else 0.0
            self.w[0] += self.lr * (outcome - self.w[0])
        return self.w
