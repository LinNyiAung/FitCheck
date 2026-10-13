"""Tape-measure style body measurements on an Anny rest-pose mesh.

Definitions (all on a standing A-pose body, feet on the floor, Z up, -Y = front):

- height:   floor to top of head.
- chest:    max horizontal taut girth between ~15 cm below the armpit and the armpit
            (bust for women, chest for men).
- waist:    min horizontal taut girth between hip and chest (natural waist).
- hip:      max horizontal taut girth between crotch and waist.
- neck:     taut girth perpendicular to the neck axis, at the neck base.
- shoulder: shoulder point to shoulder point, following the upper back surface.
- arm:      shoulder point -> elbow -> wrist (sleeve length).
- inseam:   crotch to floor.
- weight:   mesh volume x 980 kg/m^3 (Anny's convention).

"Taut" means the convex hull of the cross-section: a tape bridges hollows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from backend.body.geometry import Loop, MeshSlicer, Slice

LENGTH_KEYS = ["height", "chest", "waist", "hip", "neck", "shoulder", "arm", "inseam"]
ALL_KEYS = LENGTH_KEYS + ["weight"]
BODY_DENSITY = 980.0

COARSE_STEP = 0.01
FINE_STEP = 0.002


@dataclass
class MeasurementPlan:
    """Which mesh elements each measurement uses. Valid until the shape changes a lot."""

    loops: dict[str, Loop]
    crotch_vertex: int
    acromion: tuple[int, int]  # (left, right) vertex ids
    heights: dict[str, float] = field(default_factory=dict)  # diagnostics, metres


def _pin_ends(arc: Loop, v: np.ndarray, left: int, right: int) -> Loop:
    """Keep the part of an open left-to-right arc between two vertices, and end on them."""
    x = (v[arc.a] + arc.t[:, None] * (v[arc.b] - v[arc.a]))[:, 0]
    keep = (x < v[left, 0]) & (x > v[right, 0])
    a = np.concatenate([[left], arc.a[keep], [right]])
    b = np.concatenate([[left], arc.b[keep], [right]])
    t = np.concatenate([[0.0], arc.t[keep], [0.0]])
    return Loop(a=a, b=b, t=t, closed=False)


class BodyMeasurer:
    def __init__(self, body_model):
        self.faces = torch.as_tensor(body_model.faces, dtype=torch.long)
        self.slicer = MeshSlicer(body_model.faces)
        b = body_model.bone_index
        self.bones = {
            "neck01": b("neck01"),
            "neck02": b("neck02"),
            "neck03": b("neck03"),
            "upperarm.L": b("upperarm01.L"),
            "upperarm.R": b("upperarm01.R"),
            "elbow.L": b("lowerarm01.L"),
            "elbow.R": b("lowerarm01.R"),
            "wrist.L": b("wrist.L"),
            "wrist.R": b("wrist.R"),
        }

    # ------------------------------------------------------------------ planning (numpy)

    def _girth(self, v: np.ndarray, h: float) -> tuple[float, Slice | None]:
        s = self.slicer.torso_at(v, h)
        return (s.girth(), s) if s is not None else (np.nan, None)

    def _extremum(self, v: np.ndarray, lo: float, hi: float, find_max: bool):
        """Coarse-to-fine 1D search of torso girth over heights [lo, hi]."""
        sign = 1.0 if find_max else -1.0

        def best_of(heights):
            vals = np.array([self._girth(v, h)[0] for h in heights])
            vals = np.where(np.isnan(vals), -np.inf, sign * vals)
            return heights[int(np.argmax(vals))]

        h = best_of(np.arange(lo, hi + 1e-9, COARSE_STEP))
        h = best_of(
            np.arange(max(lo, h - COARSE_STEP), min(hi, h + COARSE_STEP) + 1e-9, FINE_STEP)
        )
        return h, self._girth(v, h)[1]

    def _half_width(self, v: np.ndarray, h: float) -> float:
        s = self.slicer.torso_at(v, h)
        return np.abs(s.points[:, 0]).max() if s is not None else np.nan

    def _armpit_height(self, v: np.ndarray, lo: float, hi: float) -> float:
        """Lowest height where the arms join the torso slice (its half-width jumps)."""
        jump = 0.03
        prev_h, prev_w = lo, self._half_width(v, lo)
        for h in np.arange(lo + COARSE_STEP, hi, COARSE_STEP):
            w = self._half_width(v, h)
            if w > prev_w + jump:
                a, b = prev_h, h
                for _ in range(8):  # bisect to ~0.05 mm
                    mid = 0.5 * (a + b)
                    if self._half_width(v, mid) > prev_w + jump:
                        b = mid
                    else:
                        a = mid
                return a
            prev_h, prev_w = h, w
        raise ValueError("Could not find the armpits: are the arms touching the torso?")

    def _acromion(self, v: np.ndarray, joint: np.ndarray) -> int:
        """Shoulder point: highest surface vertex right above the shoulder joint."""
        near = (np.abs(v[:, 0] - joint[0]) < 0.015) & (np.abs(v[:, 1] - joint[1]) < 0.04)
        near &= v[:, 2] > joint[2]
        idx = np.flatnonzero(near)
        return int(idx[np.argmax(v[idx, 2])])

    def plan(
        self,
        vertices: torch.Tensor,
        bone_heads: torch.Tensor,
        previous: MeasurementPlan | None = None,
    ) -> MeasurementPlan:
        """Locate every measurement on this mesh.

        With `previous` (a plan for a similar body, e.g. a few optimiser steps ago), the
        landmark searches are restricted to a few cm around the previous landmarks, scaled
        by the height change. That is ~4x cheaper and gives the same answer.
        """
        v = vertices.detach().cpu().numpy().astype(np.float64)
        j = bone_heads.detach().cpu().numpy().astype(np.float64)
        H = v[:, 2].max()

        def near(key, lo, hi, margin=0.03):
            """Intersect [lo, hi] with a window around the previous landmark."""
            if previous is None:
                return lo, hi
            guess = previous.heights[key] * H / previous.heights["height"]
            nlo, nhi = max(lo, guess - margin), min(hi, guess + margin)
            return (nlo, nhi) if nlo < nhi else (lo, hi)

        midline = np.flatnonzero(
            (np.abs(v[:, 0]) < 1e-4) & (v[:, 2] > 0.3 * H) & (v[:, 2] < 0.65 * H)
        )
        crotch_vertex = int(midline[np.argmin(v[midline, 2])])
        crotch = v[crotch_vertex, 2]

        armpit_lo = crotch + 0.5 * (0.8 * H - crotch)
        armpit = None
        if previous is not None:
            warm_lo = max(armpit_lo, near("armpit", 0, H, margin=0.04)[0])
            if self._half_width(v, warm_lo) < 0.5 * self._half_width(v, warm_lo + 0.08):
                armpit = self._armpit_height(v, warm_lo, 0.88 * H)
        if armpit is None:
            armpit = self._armpit_height(v, armpit_lo, 0.88 * H)
        torso = armpit - crotch

        chest_h, chest = self._extremum(
            v, *near("chest", armpit - 0.15 * H / 1.75, armpit - 0.015), True
        )
        waist_h, waist = self._extremum(
            v, *near("waist", crotch + 0.30 * torso, crotch + 0.80 * torso), False
        )
        hip_h, hip = self._extremum(v, *near("hip", crotch + 0.02, waist_h), True)

        n1, n2, n3 = (j[self.bones[k]] for k in ("neck01", "neck02", "neck03"))
        neck_origin = 0.5 * (n1 + n2)
        neck_normal = (n3 - n1) / np.linalg.norm(n3 - n1)
        neck = self.slicer.central_slice(v, neck_origin, neck_normal)

        left = self._acromion(v, j[self.bones["upperarm.L"]])
        right = self._acromion(v, j[self.bones["upperarm.R"]])
        if v[left, 0] < v[right, 0]:
            left, right = right, left

        # Shoulder width: taut tape across the upper back, pinned to the two shoulder points.
        # The shoulder top is nearly flat, so a slice's own lateral extremes jump around with
        # small height changes; slicing a little lower and clipping at the shoulder points
        # keeps the endpoints anatomical.
        shoulder_h = 0.5 * (v[left, 2] + v[right, 2]) - 0.01
        shoulder = self.slicer.torso_at(v, shoulder_h)

        slices = dict(chest=chest, waist=waist, hip=hip, neck=neck, shoulder=shoulder)
        missing = [k for k, s in slices.items() if s is None]
        if missing:
            raise ValueError(f"Could not slice: {missing}")

        return MeasurementPlan(
            loops=dict(
                chest=chest.hull_loop(),
                waist=waist.hull_loop(),
                hip=hip.hull_loop(),
                neck=neck.hull_loop(),
                shoulder=_pin_ends(shoulder.back_arc(), v, left, right),
            ),
            crotch_vertex=crotch_vertex,
            acromion=(left, right),
            heights=dict(
                height=H, crotch=crotch, armpit=armpit, chest=chest_h, waist=waist_h, hip=hip_h
            ),
        )

    # ------------------------------------------------------------- evaluation (torch)

    def volume(self, vertices: torch.Tensor) -> torch.Tensor:
        f = self.faces
        v0, v1, v2 = vertices[f[:, 0]], vertices[f[:, 1]], vertices[f[:, 2]]
        return (torch.cross(v0, v1, dim=-1) * v2).sum(-1).sum().abs() / 6.0

    def evaluate(
        self, vertices: torch.Tensor, bone_heads: torch.Tensor, plan: MeasurementPlan
    ) -> dict[str, torch.Tensor]:
        """Differentiable measurements of one body. Lengths in cm, weight in kg."""
        v, j = vertices, bone_heads
        floor = v[:, 2].min()

        def polyline(points):
            return torch.linalg.norm(points[1:] - points[:-1], dim=-1).sum()

        arms = []
        for side, acromion in zip("LR", plan.acromion):
            pts = torch.stack(
                [v[acromion], j[self.bones[f"elbow.{side}"]], j[self.bones[f"wrist.{side}"]]]
            )
            arms.append(polyline(pts))

        m = dict(
            height=v[:, 2].max() - floor,
            chest=plan.loops["chest"].length(v),
            waist=plan.loops["waist"].length(v),
            hip=plan.loops["hip"].length(v),
            neck=plan.loops["neck"].length(v),
            shoulder=plan.loops["shoulder"].length(v),
            arm=0.5 * (arms[0] + arms[1]),
            inseam=v[plan.crotch_vertex, 2] - floor,
        )
        m = {k: 100.0 * x for k, x in m.items()}
        m["weight"] = self.volume(v) * BODY_DENSITY
        return m

    def measure(self, vertices: torch.Tensor, bone_heads: torch.Tensor):
        """Exact measurement (fresh plan) of one body: (dict of floats, plan)."""
        plan = self.plan(vertices, bone_heads)
        with torch.no_grad():
            m = self.evaluate(vertices, bone_heads, plan)
        return {k: float(x) for k, x in m.items()}, plan
