"""Fit an Anny body to user-supplied measurements.

We optimise Anny's high-level phenotypes (muscle, weight, height, proportions) together with
a curated set of MakeHuman local modifiers, by gradient descent on

    sum_i ((measured_i(body) - target_i) / tolerance_i)^2  +  prior * |local changes|^2

The prior keeps local modifiers small, so the body is explained by plausible global shape
first and local tweaks only where the global shape cannot reach the numbers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from backend.body.measure import ALL_KEYS, BodyMeasurer, MeasurementPlan
from backend.body.model import ADULT_AGE, FIT_LOCAL_CHANGES, FREE_PHENOTYPES, BodyModel, fast_body

# How many cm (or kg) of error count as "one unit" in the loss. Weight from a bathroom
# scale and a volume-based mass estimate only agree loosely, so it is down-weighted.
TOLERANCE = dict(
    height=0.5, chest=1.0, waist=1.0, hip=1.0, neck=1.0, shoulder=1.0, arm=1.0, inseam=1.0,
    weight=4.0,
)
LOCAL_PRIOR = 0.5
LOCAL_RANGE = 1.0  # MakeHuman modifiers are designed for [-1, 1]

# A local modifier may only move when the user gave the measurement it exists for.
# Otherwise e.g. a weight target happily lengthens the arms to add mass. Modifiers not
# listed here are general bulk/shape controls and are always free.
LOCAL_REQUIRES = {
    "measure-bust-circ-incr": "chest",
    "measure-underbust-circ-incr": "chest",
    "breast-volume-vert-up": "chest",
    "measure-waist-circ-incr": "waist",
    "stomach-pregnant-incr": "waist",
    "measure-hips-circ-incr": "hip",
    "buttocks-volume-incr": "hip",
    "hip-scale-horiz-incr": "hip",
    "hip-scale-depth-incr": "hip",
    "measure-shoulder-dist-incr": "shoulder",
    "torso-vshape-incr": "shoulder",
    "measure-neck-circ-incr": "neck",
    "neck-scale-horiz-incr": "neck",
    "neck-scale-depth-incr": "neck",
    "measure-upperarm-length-incr": "arm",
    "measure-lowerarm-length-incr": "arm",
    "measure-upperleg-height-incr": "inseam",
    "measure-lowerleg-height-incr": "inseam",
    "measure-napetowaist-dist-incr": "inseam",
    "measure-waisttohip-dist-incr": "inseam",
}

# Stop once every length is this close (cm) and the loss has stopped improving.
CONVERGED_CM = 0.3


@dataclass
class FitResult:
    vertices: torch.Tensor  # (V, 3), metres
    bone_heads: torch.Tensor  # (J, 3)
    plan: MeasurementPlan
    targets: dict[str, float]
    achieved: dict[str, float]  # exact re-measurement of the final mesh
    phenotypes: dict[str, float]
    local_changes: dict[str, float]
    seconds: float
    iterations: int

    @property
    def errors(self) -> dict[str, float]:
        return {k: self.achieved[k] - v for k, v in self.targets.items()}


class BodyFitter:
    def __init__(self, model: BodyModel, measurer: BodyMeasurer | None = None):
        self.model = model
        self.measurer = measurer or BodyMeasurer(model)

    @staticmethod
    def _decode(raw_pheno: torch.Tensor, raw_local: torch.Tensor, local_mask: torch.Tensor):
        return torch.sigmoid(raw_pheno), LOCAL_RANGE * torch.tanh(raw_local) * local_mask

    @staticmethod
    def _converged(m: dict[str, torch.Tensor], targets: dict[str, float]) -> bool:
        for k, t in targets.items():
            limit = TOLERANCE[k] if k == "weight" else CONVERGED_CM
            if abs(float(m[k].detach()) - t) > limit:
                return False
        return True

    def fit(
        self,
        gender: float,
        targets: dict[str, float],
        iterations: int = 400,
        replan_every: int = 20,
        lr: float = 0.05,
    ) -> FitResult:
        """gender: 0 = male, 1 = female. targets: subset of ALL_KEYS (cm / kg); height required."""
        unknown = set(targets) - set(ALL_KEYS)
        if unknown:
            raise ValueError(f"Unknown measurements: {sorted(unknown)}")
        if "height" not in targets:
            raise ValueError("Height is required.")

        # Each step is a few small ops, so torch's thread pool costs far more than it saves
        # (measured 19 ms/step on 1 thread vs 700 ms on 10 when other processes are busy).
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            return self._fit(gender, targets, iterations, replan_every, lr)
        finally:
            torch.set_num_threads(threads)

    def _fit(self, gender, targets, iterations, replan_every, lr) -> FitResult:
        start = time.time()
        body = fast_body(float(gender))
        raw_pheno = torch.zeros(len(FREE_PHENOTYPES), requires_grad=True)
        raw_local = torch.zeros(len(FIT_LOCAL_CHANGES), requires_grad=True)
        target_t = {k: torch.tensor(float(x)) for k, x in targets.items()}
        local_mask = torch.tensor(
            [float(LOCAL_REQUIRES.get(k, "height") in targets) for k in FIT_LOCAL_CHANGES]
        )
        opt = torch.optim.Adam([raw_pheno, raw_local], lr=lr)

        plan = None
        for it in range(iterations):
            pheno, local = self._decode(raw_pheno, raw_local, local_mask)
            v, j = body(pheno, local)
            replan = plan is None or it % replan_every == 0
            if replan:
                plan = self.measurer.plan(v, j, previous=plan)
            m = self.measurer.evaluate(v, j, plan)
            loss = sum(((m[k] - t) / TOLERANCE[k]) ** 2 for k, t in target_t.items())
            loss = loss + LOCAL_PRIOR * (local**2).sum()
            # Right after a re-plan the measurements are exact, so this is a fair check.
            if replan and self._converged(m, targets):
                break
            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            pheno, local = self._decode(raw_pheno, raw_local, local_mask)
            v, j = body(pheno, local)
        all_pheno = dict(gender=float(gender), age=ADULT_AGE)
        all_pheno.update({k: float(x) for k, x in zip(FREE_PHENOTYPES, pheno)})
        achieved, plan = self.measurer.measure(v, j)
        return FitResult(
            vertices=v,
            bone_heads=j,
            plan=plan,
            targets=dict(targets),
            achieved=achieved,
            phenotypes=all_pheno,
            local_changes={k: float(x) for k, x in zip(FIT_LOCAL_CHANGES, local)},
            seconds=time.time() - start,
            iterations=it + 1,
        )
