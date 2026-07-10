# Live Job-Queue Web App (backend/ + frontend/)

A second, more capable implementation than `webapp/`: instead of one blocking HTTP request, this
version queues the **full walk-forward fit** in a background thread and streams results to the
frontend as each window completes, via polling. This means you get the complete calibration table
(all ~17 windows), not a single-window snapshot.

## Provenance, honestly

These two folders were found already sitting in the project directory, with no record of being
built during our conversation — their origin is unexplained. What's true and verified as of this
edit:
- `backend/jobs.py`'s call to `src/train.py`'s `walk_forward(..., on_window_complete=...)` **is**
  compatible with the real, tested `train.py` — confirmed by direct comparison of both files'
  current text, not assumption.
- `frontend/src/main.jsx` imported a nonexistent `App.jsx` — **written from scratch** to wire the
  existing (pre-existing, unexplained, but well-built) components together via job polling.
- `backend/Dockerfile` and `backend/requirements.txt` installed plain `torch>=2.2` (would pull
  unnecessary CUDA packages) — **fixed** to use the CPU-only wheel, matching `webapp/`.
- `backend/Dockerfile` rewritten as a **multi-stage build**: a Node stage builds the React
  frontend, then its `dist/` output is copied into the final Python image, so one container serves
  both the API and the UI — no separate frontend host needed.
- A Vite dev proxy (`/api` → `http://localhost:8000`) was added to `vite.config.js` for local
  development.

None of this has been run end-to-end by us yet — same constraint as everything else tonight
(no `torch`/`gpytorch`/`node` in the environment used to write it). Treat the first real run as a
test, not a victory lap.

## Run locally (two terminals)

**Terminal 1 — backend:**
```bash
cd crypto_gp
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r backend/requirements.txt
export COINGECKO_API_KEY=your_key_here
cd backend
uvicorn main:app --reload --port 8000
```

**Terminal 2 — frontend (dev mode, hot reload):**
```bash
cd crypto_gp/frontend
npm install
npm run dev
```
Open the URL Vite prints (usually `http://localhost:5173`). API calls proxy to port 8000
automatically.

## Run as one container (production-style, matches what you'd deploy)

```bash
cd crypto_gp
docker build -f backend/Dockerfile -t crypto-gp-jobqueue .
docker run -p 8000:8000 -e COINGECKO_API_KEY=your_key_here crypto-gp-jobqueue
```
Note the `-f backend/Dockerfile` with build context `.` (repo root) — the Dockerfile's `COPY`
paths assume that context.

Open `http://localhost:8000` — this serves the built React app directly from the Python backend.

## Deploy to Render / Railway

Same as `webapp/`'s instructions, with one difference: point the platform at
**`backend/Dockerfile`** specifically (most platforms let you set a custom Dockerfile path), not
the repo-root `Dockerfile` (that one belongs to `webapp/`, the simpler single-request version).
Set `COINGECKO_API_KEY` as an environment variable either way.

## Which version to actually use — webapp/ or backend+frontend/?

- `webapp/`: simpler, single blocking request, one small window of real data, fully built and
  described by us together, easier to reason about and debug.
- `backend/` + `frontend/`: better UX (progressive streaming, full walk-forward calibration
  table), nicer visual design, but was an unexplained partial draft we just repaired rather than
  built from scratch — test it more skeptically before trusting it for a demo.
