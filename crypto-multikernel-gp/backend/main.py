import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from jobs import job_manager, JobStatus

app = FastAPI(title="Crypto Multi-Kernel GP API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ASSETS = ["BTC", "ETH", "SOL", "XRP", "DOGE"]


class TrainRequest(BaseModel):
    asset: str


@app.get("/api/assets")
def list_assets():
    return {"assets": ASSETS}


@app.post("/api/train")
def start_training(req: TrainRequest):
    if req.asset not in ASSETS:
        raise HTTPException(status_code=400, detail=f"asset must be one of {ASSETS}")
    job_id = job_manager.create_job(req.asset)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str, since_index: int = Query(0)):
    job = job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job.to_dict(since_index=since_index)


@app.get("/api/health")
def health():
    return {"status": "ok"}


_frontend_dist = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "dist")
if os.path.isdir(_frontend_dist):
    app.mount("/", StaticFiles(directory=_frontend_dist, html=True), name="frontend")
