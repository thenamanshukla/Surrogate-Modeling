import threading
import uuid
import time
import traceback
from enum import Enum


class JobStatus(str, Enum):
    QUEUED = "queued"
    FETCHING_DATA = "fetching_data"
    ENGINEERING_FEATURES = "engineering_features"
    TRAINING = "training"
    DONE = "done"
    ERROR = "error"


class Job:
    def __init__(self, job_id, asset):
        self.job_id = job_id
        self.asset = asset
        self.status = JobStatus.QUEUED
        self.progress = 0.0
        self.windows_done = 0
        self.windows_total = 0
        self.records = []
        self.calibration_table = None
        self.point_metrics = None
        self.error = None
        self.created_at = time.time()
        self.lock = threading.Lock()

    def to_dict(self, since_index=0):
        with self.lock:
            new_records = self.records[since_index:]
            return {
                "job_id": self.job_id,
                "asset": self.asset,
                "status": self.status,
                "progress": self.progress,
                "windows_done": self.windows_done,
                "windows_total": self.windows_total,
                "new_records": new_records,
                "total_records": len(self.records),
                "calibration_table": self.calibration_table,
                "point_metrics": self.point_metrics,
                "error": self.error,
            }


class JobManager:
    def __init__(self):
        self.jobs = {}
        self.lock = threading.Lock()

    def create_job(self, asset):
        job_id = str(uuid.uuid4())
        job = Job(job_id, asset)
        with self.lock:
            self.jobs[job_id] = job
        thread = threading.Thread(target=self._run_job, args=(job,), daemon=True)
        thread.start()
        return job_id

    def get_job(self, job_id):
        with self.lock:
            return self.jobs.get(job_id)

    def _run_job(self, job):
        try:
            import os
            import sys

            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from src.features import build_dataset
            from src import train as train_module
            from src.evaluate import point_metrics, calibration_table

            data_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
            )
            chart_path = os.path.join(data_dir, f"{job.asset}_market_chart.csv")

            job.status = JobStatus.FETCHING_DATA
            if not os.path.exists(chart_path):
                from src.data_acquisition import (
                    fetch_market_chart,
                    fetch_coin_snapshot,
                    fetch_fear_greed_index,
                    ASSETS,
                )
                import json

                os.makedirs(data_dir, exist_ok=True)
                fng_path = os.path.join(data_dir, "fear_greed_index.csv")
                if not os.path.exists(fng_path):
                    fng = fetch_fear_greed_index(limit=0)
                    fng.to_csv(fng_path)

                coin_id = ASSETS[job.asset]
                chart = fetch_market_chart(coin_id, days=365)
                chart.to_csv(chart_path)
                snapshot = fetch_coin_snapshot(coin_id)
                with open(os.path.join(data_dir, f"{job.asset}_snapshot.json"), "w") as fh:
                    json.dump(snapshot, fh)

            job.status = JobStatus.ENGINEERING_FEATURES
            build_dataset(job.asset)

            job.status = JobStatus.TRAINING

            def on_window_complete(window_records, window_count, total_windows):
                with job.lock:
                    job.records.extend(window_records)
                    job.windows_done = window_count
                    job.windows_total = total_windows
                    job.progress = min(99.0, 100.0 * window_count / max(1, total_windows))

            train_module.walk_forward(job.asset, on_window_complete=on_window_complete)

            import pandas as pd

            with job.lock:
                df = pd.DataFrame(job.records)

            if len(df) > 0:
                metrics = point_metrics(df)
                calib = calibration_table(df)
                with job.lock:
                    job.point_metrics = {k: float(v) for k, v in metrics.items()}
                    job.calibration_table = calib.to_dict(orient="records")

            with job.lock:
                job.status = JobStatus.DONE
                job.progress = 100.0

        except Exception as e:
            with job.lock:
                job.status = JobStatus.ERROR
                job.error = f"{str(e)}\n{traceback.format_exc()}"


job_manager = JobManager()
