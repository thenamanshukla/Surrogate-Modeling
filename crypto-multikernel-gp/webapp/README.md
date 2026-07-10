# Live Web App

A FastAPI backend that trains the multi-kernel additive GP **live, on request** for a chosen
asset, plus a static single-page frontend that visualizes the per-modality uncertainty
decomposition, calibration, and predictions.

## What "live" actually means here

Each `POST /api/train/{asset}` call: re-fetches market/dev-activity/sentiment data if it's more
than 6 hours stale, rebuilds features, fits a fresh GP via SVI with validation-based early
stopping on the last ~180 days of real data, and returns predictions + the exact variance
decomposition for the most recent ~14 held-out days. This is a single train/test split, not the
full walk-forward loop from the research pipeline (`src/train.py`) — that would be too slow for
an interactive request. Expect the request to take anywhere from ~10 seconds to ~2 minutes
depending on host CPU.

## Environment variable required

```
COINGECKO_API_KEY=your_free_demo_key
```
Get one at https://www.coingecko.com/en/api/pricing under "Start with our Demo Plan."

## Run locally

```bash
cd crypto_gp
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r webapp/requirements.txt
export COINGECKO_API_KEY=your_key_here
export PYTHONPATH=.
uvicorn webapp.backend.main:app --reload --port 8000
```
Open http://localhost:8000

## Deploy with Docker (Render, Railway, Fly.io, a plain VM)

```bash
docker build -t crypto-gp-webapp .
docker run -p 8000:8000 -e COINGECKO_API_KEY=your_key_here crypto-gp-webapp
```

### Render
1. New → Web Service → connect this repo (must include the `Dockerfile` at the repo root).
2. Render auto-detects the Dockerfile. Set the environment variable `COINGECKO_API_KEY`.
3. Pick at least the smallest **paid** instance if available — the free tier's spin-down/cold-start
   plus CPU limits make a live GP fit (even with the small encoders and early stopping we tuned)
   uncomfortably slow or may hit request timeouts. 512MB-1GB RAM is enough; this is a CPU-bound
   workload, no GPU needed.

### Railway
1. New Project → Deploy from repo → Railway detects the Dockerfile automatically.
2. Add `COINGECKO_API_KEY` under Variables.
3. Railway's default free/hobby tier CPU is generally sufficient for this workload's size.

### Plain VM
```bash
git clone <your repo>
cd crypto_gp
docker build -t crypto-gp-webapp .
docker run -d -p 80:8000 -e COINGECKO_API_KEY=your_key_here --restart unless-stopped crypto-gp-webapp
```

## Known limitations of this live version, honestly

- CoinGecko's free tier caps history at 365 days, so every asset trains on well under a year of
  data — see the main README for why this matters for statistical confidence.
- The 6-hour data cache is file-based and per-instance; if your host scales to multiple instances,
  each will independently re-fetch data, which is fine at this traffic scale but worth knowing.
- No queueing: if two people click "fit" at once, both requests train independently and
  concurrently on the same CPU, competing for resources. Fine for a demo, not for production
  traffic — add a job queue (e.g. Celery/RQ) before that becomes a real concern.
- The Keras LSTM and GARCH baselines from the research pipeline are intentionally **not** wired
  into this live endpoint, to keep the Docker image and cold-start time small. They still exist
  in `src/models/baselines.py` for offline comparison.
