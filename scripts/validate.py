"""Round-trip validation of the body fitter.

For N random synthetic bodies: measure them, fit a new body to those measurements only,
then report (a) how well the measurements are reproduced and (b) how close the recovered
3D shape is to the original (mean / 95th percentile vertex distance).

(b) tells us how much 9 tape measurements actually pin down a body. Shape that the
measurements don't constrain (posture, belly vs. back fat, ...) shows up here.

Usage:  .venv\\Scripts\\python scripts\\validate.py [N]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.body.fit import BodyFitter  # noqa: E402
from backend.body.measure import ALL_KEYS  # noqa: E402
from backend.body.model import FIT_LOCAL_CHANGES, fast_body, load_body_model  # noqa: E402


def random_body(rng: np.random.Generator, gender: float):
    pheno = torch.tensor(rng.uniform(0.15, 0.85, 4), dtype=torch.float32)
    local = torch.tensor(rng.uniform(-0.4, 0.4, len(FIT_LOCAL_CHANGES)), dtype=torch.float32)
    with torch.no_grad():
        return fast_body(gender)(pheno, local)


def main(n: int = 10, seed: int = 0) -> None:
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed)
    model = load_body_model()
    fitter = BodyFitter(model)

    meas_err = {k: [] for k in ALL_KEYS}
    shape_mean, shape_p95, seconds = [], [], []
    for i in range(n):
        gender = float(i % 2)
        v_true, j_true = random_body(rng, gender)
        targets, _ = fitter.measurer.measure(v_true, j_true)
        t = time.time()
        r = fitter.fit(gender, targets)
        seconds.append(time.time() - t)
        for k in ALL_KEYS:
            meas_err[k].append(abs(r.achieved[k] - targets[k]))
        d = torch.linalg.norm(r.vertices - v_true, dim=-1).numpy() * 1000  # mm
        shape_mean.append(d.mean())
        shape_p95.append(np.percentile(d, 95))
        print(
            f"body {i:2d} ({'F' if gender else 'M'}, {targets['height']:.0f} cm): "
            f"worst measurement error {max(abs(r.achieved[k] - targets[k]) for k in ALL_KEYS if k != 'weight'):.2f} cm, "
            f"shape error mean {d.mean():.1f} mm / p95 {np.percentile(d, 95):.1f} mm, "
            f"{seconds[-1]:.1f}s"
        )

    print("\nMeasurement reproduction (|fitted - true|):")
    for k in ALL_KEYS:
        e = np.array(meas_err[k])
        unit = "kg" if k == "weight" else "cm"
        print(f"  {k:9s} mean {e.mean():.2f} {unit}   max {e.max():.2f} {unit}")
    print(
        f"\n3D shape recovery: mean vertex error {np.mean(shape_mean):.1f} mm, "
        f"p95 {np.mean(shape_p95):.1f} mm"
    )
    print(f"Fit time: mean {np.mean(seconds):.1f}s, max {np.max(seconds):.1f}s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
