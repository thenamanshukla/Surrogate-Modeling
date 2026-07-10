import os
import sys
import time
import json
import numpy as np
import pandas as pd
import torch
import gpytorch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.features import build_dataset
from src.data_acquisition import (
    ASSETS,
    fetch_market_chart,
    fetch_coin_snapshot,
    fetch_fear_greed_index,
    DATA_DIR,
)
from src.models.multi_kernel_gp import MultiKernelAdditiveGP, ModalitySlices
from src.train import train_gp, standardize
from src.models.decomposition import exact_variance_decomposition, dominant_modality
from src.evaluate import point_metrics as compute_point_metrics
from src.evaluate import coverage as compute_coverage

DATA_MAX_AGE_SECONDS = 6 * 3600


def _file_is_fresh(path):
    if not os.path.exists(path):
        return False
    return (time.time() - os.path.getmtime(path)) < DATA_MAX_AGE_SECONDS


def ensure_data(asset, force=False):
    if asset not in ASSETS:
        raise ValueError(f"unknown asset {asset}")
    coin_id = ASSETS[asset]

    fng_path = os.path.join(DATA_DIR, "fear_greed_index.csv")
    if force or not _file_is_fresh(fng_path):
        fng = fetch_fear_greed_index(limit=0)
        fng.to_csv(fng_path)

    chart_path = os.path.join(DATA_DIR, f"{asset}_market_chart.csv")
    snapshot_path = os.path.join(DATA_DIR, f"{asset}_snapshot.json")
    if force or not _file_is_fresh(chart_path):
        chart = fetch_market_chart(coin_id, days=365)
        chart.to_csv(chart_path)
    if force or not _file_is_fresh(snapshot_path):
        snapshot = fetch_coin_snapshot(coin_id)
        with open(snapshot_path, "w") as fh:
            json.dump(snapshot, fh)


def assemble_tensor(d):
    X = pd.concat([d["fundamental"], d["technical"], d["sentiment"], d["regime"]], axis=1)
    slices = ModalitySlices(d["fundamental"].shape[1], d["technical"].shape[1], d["sentiment"].shape[1])
    y = d["target"]
    return X, y, slices


def run_live_pipeline(asset, window_train=180, horizon_test=14, progress_cb=None):
    def report(msg):
        if progress_cb:
            progress_cb(msg)

    report("fetching live market, on-chain, and sentiment data")
    ensure_data(asset)

    report("engineering fundamental, technical, and sentiment features")
    d = build_dataset(asset)
    X, y, slices = assemble_tensor(d)

    if len(X) < window_train + horizon_test:
        window_train = max(int(len(X) * 0.7), 30)
        horizon_test = max(len(X) - window_train, 5)

    train_X_raw = X.iloc[-(window_train + horizon_test) : -horizon_test]
    test_X_raw = X.iloc[-horizon_test:]
    train_y = torch.tensor(y.iloc[-(window_train + horizon_test) : -horizon_test].values, dtype=torch.float32)
    test_y = torch.tensor(y.iloc[-horizon_test:].values, dtype=torch.float32)

    train_Xs, test_Xs = standardize(train_X_raw, test_X_raw)
    train_x = torch.tensor(train_Xs.values, dtype=torch.float32)
    test_x = torch.tensor(test_Xs.values, dtype=torch.float32)

    y_mean = train_y.mean()
    y_std = train_y.std()
    train_y_scaled = (train_y - y_mean) / y_std

    report("fitting multi-kernel additive Gaussian process (SVI)")
    model, likelihood = train_gp(train_x, train_y_scaled, slices, verbose=False)
    model.eval()
    likelihood.eval()

    report("computing predictive distribution and uncertainty decomposition")
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = likelihood(model(test_x))
        mean_raw = posterior.mean.numpy()
        var_raw = posterior.variance.numpy()

    n_test = test_x.shape[0]

    def reduce_to_per_point(arr, n_test):
        arr = np.asarray(arr)
        if arr.ndim == 1:
            if arr.shape[0] == n_test:
                return arr
            return np.repeat(arr.mean(), n_test)
        match_axes = [ax for ax, sz in enumerate(arr.shape) if sz == n_test]
        if match_axes:
            axis = match_axes[0]
            other_axes = tuple(a for a in range(arr.ndim) if a != axis)
            return arr.mean(axis=other_axes) if other_axes else arr
        return arr.reshape(n_test, -1).mean(axis=1)

    pred_mean = reduce_to_per_point(mean_raw, n_test) * y_std.item() + y_mean.item()
    pred_var = reduce_to_per_point(var_raw, n_test) * (y_std.item() ** 2)

    noise_var = likelihood.noise.item() if hasattr(likelihood, "noise") else 1e-3
    decomp = exact_variance_decomposition(model, train_x, test_x, noise_var)
    dominant = dominant_modality(decomp["diagonal"])

    dof = 4.0
    results_df = pd.DataFrame({
        "date": test_X_raw.index,
        "y_true": test_y.numpy(),
        "pred_mean": pred_mean,
        "pred_var": pred_var,
        "var_fundamental": decomp["diagonal"]["fundamental"],
        "var_technical": decomp["diagonal"]["technical"],
        "var_sentiment": decomp["diagonal"]["sentiment"],
        "var_interaction": decomp["interaction"],
        "dominant_modality": dominant,
    })

    metrics = compute_point_metrics(results_df)
    calibration = []
    for level in (0.5, 0.8, 0.9, 0.95):
        calibration.append({"level": level, "coverage": float(compute_coverage(results_df, level, dof=dof))})

    report("done")

    predictions = []
    q95 = 2.776
    for _, row in results_df.iterrows():
        scale = np.sqrt(max(row["pred_var"], 0.0) * (dof - 2) / dof)
        predictions.append({
            "date": row["date"].strftime("%Y-%m-%d"),
            "y_true": float(row["y_true"]),
            "pred_mean": float(row["pred_mean"]),
            "pred_lower_95": float(row["pred_mean"] - q95 * scale),
            "pred_upper_95": float(row["pred_mean"] + q95 * scale),
            "var_fundamental": float(row["var_fundamental"]),
            "var_technical": float(row["var_technical"]),
            "var_sentiment": float(row["var_sentiment"]),
            "var_interaction": float(row["var_interaction"]),
            "dominant_modality": row["dominant_modality"],
        })

    return {
        "asset": asset,
        "window_train": window_train,
        "horizon_test": horizon_test,
        "dof": dof,
        "metrics": {
            "rmse": float(metrics["rmse"]),
            "mae": float(metrics["mae"]),
            "directional_accuracy": float(metrics["directional_accuracy"]),
        },
        "calibration": calibration,
        "predictions": predictions,
        "caveat": (
            f"Trained live on the last {window_train} days of real market data "
            f"(free-tier history limit), evaluated on the most recent {horizon_test} "
            "held-out days. Small test size means calibration numbers are indicative, "
            "not statistically definitive."
        ),
    }
