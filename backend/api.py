"""HTTP API for the body fitting prototype; also serves the built React UI.

Run:  .venv\\Scripts\\python -m uvicorn backend.api:app --port 8000
UI:   cd frontend && npm run build   (or `npm run dev` for hot reload on :5173)
"""

from __future__ import annotations

import base64
import threading
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.body.fit import BodyFitter, FitResult
from backend.body.model import fast_body, load_body_model

UI_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"

LABELS = dict(
    height="Height", chest="Chest / bust", waist="Waist", hip="Hip", neck="Neck",
    shoulder="Shoulder width", arm="Arm length", inseam="Inseam", weight="Weight",
)
# A fitted value further than this from the input is reported as "not matched".
MATCH_TOLERANCE = dict(weight=4.0)
DEFAULT_MATCH_TOLERANCE = 1.0


class FitRequest(BaseModel):
    gender: Literal["male", "female"]
    height: float = Field(ge=140, le=215, description="cm")
    chest: float | None = Field(default=None, ge=60, le=170)
    waist: float | None = Field(default=None, ge=50, le=170)
    hip: float | None = Field(default=None, ge=60, le=180)
    neck: float | None = Field(default=None, ge=25, le=60)
    shoulder: float | None = Field(default=None, ge=28, le=65)
    arm: float | None = Field(default=None, ge=40, le=85)
    inseam: float | None = Field(default=None, ge=55, le=105)
    weight: float | None = Field(default=None, ge=35, le=200, description="kg")

    def targets(self) -> dict[str, float]:
        return {
            k: v for k, v in self.model_dump(exclude={"gender"}).items() if v is not None
        }


def plausibility_warnings(r: FitRequest) -> list[str]:
    """Cheap checks for typos and measuring mistakes, before we spend seconds fitting.

    Ratios are generous adult ranges; a proper version would score inputs against an
    anthropometric survey such as ANSUR II.
    """
    w, h = [], r.height
    if r.inseam and not 0.38 <= r.inseam / h <= 0.53:
        w.append(f"Inseam is {r.inseam / h:.0%} of height (typical 42-49%). Measure crotch to floor, no shoes.")
    if r.arm and not 0.29 <= r.arm / h <= 0.40:
        w.append(f"Arm length is {r.arm / h:.0%} of height (typical 32-37%). Measure shoulder tip to wrist bone.")
    if r.weight:
        bmi = r.weight / (h / 100) ** 2
        if not 15 <= bmi <= 45:
            w.append(f"Height and weight give a BMI of {bmi:.0f}, which is unusual. Please double-check both.")
    if r.waist and r.hip and r.waist > 1.3 * r.hip:
        w.append("Waist is much larger than hip. Measure the hip at the fullest part of the buttocks.")
    if r.chest and r.waist and r.waist > 1.35 * r.chest:
        w.append("Waist is much larger than chest. Measure the chest under the armpits, at the fullest part.")
    if r.neck and r.chest and not 0.28 <= r.neck / r.chest <= 0.50:
        w.append("Neck and chest don't look consistent. Measure the neck at its base, where a collar sits.")
    return w


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def _to_y_up(p: np.ndarray) -> np.ndarray:
    """Anny is Z-up with -Y as the front; three.js is Y-up with +Z towards the viewer."""
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], axis=-1)


def tape_lines(result: FitResult, fitter: BodyFitter) -> dict[str, list[list[list[float]]]]:
    """Polylines showing where each measurement was taken, for drawing on the body."""
    v, j, plan = result.vertices, result.bone_heads, result.plan
    bones = fitter.measurer.bones
    lines: dict[str, list[np.ndarray]] = {}
    with torch.no_grad():
        for key, loop in plan.loops.items():
            pts = loop.points(v).numpy()
            lines[key] = [np.vstack([pts, pts[:1]]) if loop.closed else pts]
        lines["arm"] = [
            np.stack([v[a].numpy(), j[bones[f"elbow.{s}"]].numpy(), j[bones[f"wrist.{s}"]].numpy()])
            for s, a in zip("LR", plan.acromion)
        ]
        crotch = v[plan.crotch_vertex].numpy()
        lines["inseam"] = [np.stack([crotch, crotch * np.array([1, 1, 0])])]
        top = v[int(torch.argmax(v[:, 2]))].numpy()
        side = np.array([v[:, 0].max().item() + 0.08, top[1], 0.0])
        lines["height"] = [np.stack([side, side + np.array([0, 0, top[2]])])]
    return {k: [_to_y_up(p).round(4).tolist() for p in ps] for k, ps in lines.items()}


app = FastAPI(title="FitCheck body prototype")
_fit_lock = threading.Lock()  # the fitter is CPU-bound and single-threaded by design


@app.on_event("startup")
def _warm_up() -> None:
    load_body_model()
    fast_body(0.0)
    fast_body(1.0)


@app.get("/")
def index():
    page = UI_DIR / "index.html"
    if not page.exists():
        return PlainTextResponse(
            "UI not built. Run `npm install` and `npm run build` in frontend/, "
            "or use `npm run dev` and open http://localhost:5173.",
            status_code=503,
        )
    return FileResponse(page)


if (UI_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=UI_DIR / "assets"), name="assets")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.post("/api/fit")
def fit(req: FitRequest) -> dict:
    model = load_body_model()
    fitter = BodyFitter(model)
    gender = 0.0 if req.gender == "male" else 1.0
    try:
        with _fit_lock:
            result = fitter.fit(gender, req.targets())
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    rows, unmatched = [], []
    for key, label in LABELS.items():
        achieved = result.achieved[key]
        target = result.targets.get(key)
        row = dict(key=key, label=label, unit="kg" if key == "weight" else "cm",
                   target=target, achieved=round(achieved, 1))
        if target is not None:
            row["error"] = round(achieved - target, 1)
            if abs(achieved - target) > MATCH_TOLERANCE.get(key, DEFAULT_MATCH_TOLERANCE):
                unmatched.append(label.lower())
        rows.append(row)

    warnings = plausibility_warnings(req)
    if unmatched:
        warnings.append(
            "The body could not match " + ", ".join(unmatched)
            + " exactly. This combination is rare, so please re-check those numbers."
        )

    verts = _to_y_up(result.vertices.numpy()).astype(np.float32)
    return dict(
        measurements=rows,
        warnings=warnings,
        seconds=round(result.seconds, 2),
        iterations=result.iterations,
        shape=dict(
            phenotypes={k: round(x, 3) for k, x in result.phenotypes.items()},
            local_changes={k: round(x, 3) for k, x in result.local_changes.items() if abs(x) > 0.01},
        ),
        mesh=dict(
            positions=_b64(verts),
            indices=_b64(model.faces.astype(np.uint32)),
        ),
        tapes=tape_lines(result, fitter),
    )
