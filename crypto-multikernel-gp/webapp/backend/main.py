import os
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .pipeline import run_live_pipeline, ASSETS

app = FastAPI(title="Crypto Uncertainty Decomposition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/assets")
def list_assets():
    return {"assets": list(ASSETS.keys())}


@app.post("/api/train/{asset}")
def train(asset: str):
    asset = asset.upper()
    if asset not in ASSETS:
        raise HTTPException(status_code=400, detail=f"unsupported asset '{asset}'. choices: {list(ASSETS.keys())}")
    try:
        result = run_live_pipeline(asset)
        return result
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
