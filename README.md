# FitCheck: body model prototype

Step 1 of the virtual try-on system: turn a user's tape measurements into a 3D body that
has exactly those measurements.

## Run it

One-time setup (Python backend + React UI):

```bash
python -m venv .venv
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\pip install -r requirements.txt
cd frontend && npm install && npm run build
```

Then start the server:

```bash
.venv\Scripts\python -m uvicorn backend.api:app --port 8000
```

Open http://localhost:8000, choose a gender, enter measurements, then click **Generate my body**.

**Working on the UI:** keep the backend running, and in a second terminal run
`npm run dev` inside `frontend/`. Then open http://localhost:5173. Changes to the React code
show up instantly, and `/api` calls are forwarded to the backend on port 8000. Run
`npm run build` again when you're done, so port 8000 serves the new version.

The first launch on a new machine parses the body model assets and caches them in
`~/.cache/anny`, which takes about 10 minutes. Later launches start in seconds.

## How it works

| Step | File | What it does |
|---|---|---|
| Body model | [backend/body/model.py](backend/body/model.py) | [Anny](https://github.com/naver/anny) (Apache-2.0, built on MakeHuman's CC0 assets) is a differentiable parametric body. `FastBody` re-expresses it exactly for one gender as a single matrix product, about 100x faster than Anny's generic path. |
| Measuring | [backend/body/measure.py](backend/body/measure.py), [geometry.py](backend/body/geometry.py) | Measures the mesh the way a tape does. Girths are the convex hull of a plane slice, landmarks (armpit, crotch, natural waist, fullest hip) are found automatically, and shoulder width follows the upper back. Every measurement is differentiable. |
| Fitting | [backend/body/fit.py](backend/body/fit.py) | Adam optimizes 4 global shape parameters (muscle, weight, height, proportions) and 24 MakeHuman measurement modifiers to minimize the measurement error. A prior keeps local modifiers small, and a modifier only moves when its measurement was given. |
| API | [backend/api.py](backend/api.py) | `POST /api/fit` returns the mesh, a target-vs-achieved table, plausibility warnings and the tape-line positions. Also serves the built UI from `frontend/dist`. |
| UI | [frontend/src](frontend/src) | React + TypeScript (Vite). The 3D view uses React Three Fiber and drei: [BodyViewer.tsx](frontend/src/components/BodyViewer.tsx) draws the body and tape lines. The form, results table and GLB export are in [MeasurementForm.tsx](frontend/src/components/MeasurementForm.tsx), [ResultsPanel.tsx](frontend/src/components/ResultsPanel.tsx) and [App.tsx](frontend/src/App.tsx). |

### Measurement definitions

All measured on a standing A-pose body:

- **height**: floor to the top of the head
- **chest / bust**: fullest horizontal girth below the armpits
- **waist**: narrowest horizontal girth (natural waist)
- **hip**: fullest horizontal girth above the crotch
- **neck**: girth at the neck base, perpendicular to the neck
- **shoulder**: from one shoulder tip to the other, across the back
- **arm**: shoulder tip to elbow to wrist
- **inseam**: crotch to floor
- **weight**: mesh volume × 980 kg/m³

These must match how merchants measure garments. Agree on them before building the garment step.

## Validation

```bash
.venv\Scripts\python scripts\validate.py 10
```

This generates random bodies, measures them, fits new bodies from those measurements only,
and compares the two. Latest run on 10 bodies:

- Measurements reproduced within **0.3 cm** on average (worst single case: shoulder, 2.4 cm). Weight within 2 kg.
- Recovered 3D shape: **13 mm** mean vertex distance from the true body. This is the part
  9 tape measurements can't pin down, such as how the same waist girth is split between belly and back.
- Fit time 3–18 s on a laptop CPU, single-threaded. The fitter deliberately uses one
  thread, because torch's thread pool made each step about 35x slower when other programs were busy.

## Known limitations / next steps

- **Weight is approximate.** Anny's volume-based mass runs a few kg light compared with
  real scales, so weight is a loose hint (±4 kg).
- **Height-only input gives Anny's "average" body at that height**, not a population average.
  The next step is priors from a real anthropometric survey (ANSUR II): fill in missing
  measurements statistically and score how plausible the inputs are.
- **Shoulder points come from a heuristic** (the highest surface point above the shoulder joint). It's
  the least stable measurement.
- Adults only, fixed age, fixed A-pose. Posture differences (sway back, forward head) aren't modeled.
- No real-person validation yet. The next validation step is comparing against people
  measured by hand, or against a public dataset such as BodyM.
