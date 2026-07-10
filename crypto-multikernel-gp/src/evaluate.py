import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


def fit_variance_recalibration_scale(df, dof=4.0):
    residual_var = np.mean((df["y_true"] - df["pred_mean"]) ** 2)
    raw_predictive_var = np.mean(df["pred_var"] * (dof - 2) / dof)
    scale = residual_var / raw_predictive_var
    return scale


def apply_variance_recalibration(df, scale):
    calibrated = df.copy()
    calibrated["pred_var"] = df["pred_var"] * scale
    for col in ["var_fundamental", "var_technical", "var_sentiment", "var_interaction"]:
        if col in calibrated.columns:
            calibrated[col] = df[col] * scale
    return calibrated


def point_metrics(df):
    err = df["y_true"] - df["pred_mean"]
    rmse = np.sqrt(np.mean(err**2))
    mae = np.mean(np.abs(err))
    directional_acc = np.mean(np.sign(df["y_true"]) == np.sign(df["pred_mean"]))
    return {"rmse": rmse, "mae": mae, "directional_accuracy": directional_acc}


def coverage(df, level, dof=4.0):
    alpha = 1 - level
    q = stats.t.ppf(1 - alpha / 2, df=dof)
    scale = np.sqrt(df["pred_var"] * (dof - 2) / dof)
    lower = df["pred_mean"] - q * scale
    upper = df["pred_mean"] + q * scale
    within = (df["y_true"] >= lower) & (df["y_true"] <= upper)
    return within.mean()


def coverage_by_dominant_modality(df, level, dof=4.0):
    results = {}
    for modality in df["dominant_modality"].unique():
        subset = df[df["dominant_modality"] == modality]
        results[modality] = coverage(subset, level, dof)
    return results


def crps_student_t(y_true, mean, var, dof=4.0, n_samples=2000, seed=0):
    rng = np.random.default_rng(seed)
    scale = np.sqrt(var * (dof - 2) / dof)
    samples = mean[:, None] + scale[:, None] * rng.standard_t(dof, size=(len(mean), n_samples))
    term1 = np.mean(np.abs(samples - y_true[:, None]), axis=1)
    term2 = np.mean(
        np.abs(samples[:, :, None] - samples[:, None, :]).mean(axis=(1, 2))
    )
    return term1 - 0.5 * term2


def negative_log_predictive_density(y_true, mean, var, dof=4.0):
    scale = np.sqrt(var * (dof - 2) / dof)
    logpdf = stats.t.logpdf(y_true, df=dof, loc=mean, scale=scale)
    return -np.mean(logpdf)


def calibration_table(df, levels=(0.5, 0.8, 0.9, 0.95)):
    rows = []
    for level in levels:
        overall = coverage(df, level)
        by_modality = coverage_by_dominant_modality(df, level)
        row = {"nominal_level": level, "empirical_coverage_overall": overall}
        row.update({f"coverage_when_{k}_dominant": v for k, v in by_modality.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def plot_variance_decomposition(df, asset, save=True):
    subset = df[df["asset"] == asset].sort_values("date")
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.stackplot(
        subset["date"],
        subset["var_fundamental"],
        subset["var_technical"],
        subset["var_sentiment"],
        labels=["fundamental", "technical", "sentiment"],
        alpha=0.85,
    )
    ax.plot(subset["date"], subset["var_interaction"], color="black", linestyle="--", label="interaction")
    ax.legend(loc="upper left")
    ax.set_title(f"{asset}: per-modality predictive variance decomposition")
    ax.set_ylabel("variance contribution")
    fig.autofmt_xdate()
    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, f"{asset}_variance_decomposition.png"), dpi=150, bbox_inches="tight")
    return fig


def regime_event_check(df, events_csv):
    events = pd.read_csv(events_csv, parse_dates=["date"])
    rows = []
    for _, ev in events.iterrows():
        window = df[
            (df["asset"] == ev["asset"])
            & (df["date"] >= ev["date"] - pd.Timedelta(days=3))
            & (df["date"] <= ev["date"] + pd.Timedelta(days=3))
        ]
        if window.empty:
            continue
        baseline = df[
            (df["asset"] == ev["asset"])
            & (df["date"] < ev["date"] - pd.Timedelta(days=30))
        ]
        rows.append(
            {
                "asset": ev["asset"],
                "event": ev["description"],
                "date": ev["date"],
                "sentiment_var_during_event": window["var_sentiment"].mean(),
                "sentiment_var_baseline": baseline["var_sentiment"].mean() if not baseline.empty else np.nan,
                "fundamental_var_during_event": window["var_fundamental"].mean(),
                "fundamental_var_baseline": baseline["var_fundamental"].mean() if not baseline.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def full_report(combined_results_csv):
    df = pd.read_csv(combined_results_csv, parse_dates=["date"])
    scale = fit_variance_recalibration_scale(df)
    print(f"variance recalibration scale: {scale:.4f}")
    df_calibrated = apply_variance_recalibration(df, scale)
    df_calibrated.to_csv(os.path.join(OUTPUT_DIR, "all_assets_walk_forward_results_calibrated.csv"), index=False)

    metrics = point_metrics(df_calibrated)
    calib = calibration_table(df_calibrated)
    print(metrics)
    print(calib)
    calib.to_csv(os.path.join(OUTPUT_DIR, "calibration_table.csv"), index=False)
    for asset in df_calibrated["asset"].unique():
        plot_variance_decomposition(df_calibrated, asset)
    return metrics, calib


if __name__ == "__main__":
    full_report(os.path.join(OUTPUT_DIR, "all_assets_walk_forward_results.csv"))
