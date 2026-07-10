import os
import numpy as np
import pandas as pd
import torch
import gpytorch

from .features import build_dataset
from .models.multi_kernel_gp import MultiKernelAdditiveGP, ModalitySlices, build_likelihood
from .models.decomposition import exact_variance_decomposition, dominant_modality

ASSETS = ["BTC", "ETH", "SOL", "XRP", "DOGE"]
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def assemble_tensor(d):
    X = pd.concat([d["fundamental"], d["technical"], d["sentiment"], d["regime"]], axis=1)
    slices = ModalitySlices(d["fundamental"].shape[1], d["technical"].shape[1], d["sentiment"].shape[1])
    y = d["target"]
    return X, y, slices


def standardize(train_X, test_X):
    mu = train_X.mean(axis=0)
    sigma = train_X.std(axis=0).replace(0, 1.0)
    return (train_X - mu) / sigma, (test_X - mu) / sigma


def train_gp(train_x, train_y, slices, n_iter=600, lr=0.02, n_inducing=48, verbose=False, weight_decay=5e-3, patience=40, val_frac=0.2):
    n = train_x.shape[0]
    n_val = max(int(n * val_frac), 10)
    perm = torch.randperm(n)
    val_idx = perm[:n_val]
    fit_idx = perm[n_val:]

    fit_x, fit_y = train_x[fit_idx], train_y[fit_idx]
    val_x, val_y = train_x[val_idx], train_y[val_idx]

    inducing_idx = np.random.choice(fit_x.shape[0], size=min(n_inducing, fit_x.shape[0]), replace=False)
    inducing_points = fit_x[inducing_idx].clone()

    model = MultiKernelAdditiveGP(inducing_points, slices)
    likelihood = build_likelihood()

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(likelihood.parameters()), lr=lr, weight_decay=weight_decay
    )
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=fit_y.size(0))

    best_val_nll = float("inf")
    best_state = None
    stall_count = 0

    for i in range(n_iter):
        model.train()
        likelihood.train()
        optimizer.zero_grad()
        output = model(fit_x)
        loss = -mll(output, fit_y)
        loss.backward()
        optimizer.step()

        model.eval()
        likelihood.eval()
        with torch.no_grad():
            val_posterior = likelihood(model(val_x))
            val_nll = -val_posterior.log_prob(val_y).mean().item()

        if verbose and i % 50 == 0:
            print(f"iter {i}: train_loss {loss.item():.4f} val_nll {val_nll:.4f}")

        if val_nll < best_val_nll - 1e-3:
            best_val_nll = val_nll
            best_state = {
                "model": {k: v.clone() for k, v in model.state_dict().items()},
                "likelihood": {k: v.clone() for k, v in likelihood.state_dict().items()},
            }
            stall_count = 0
        else:
            stall_count += 1
        if stall_count >= patience:
            if verbose:
                print(f"early stop at iter {i}, best val_nll {best_val_nll:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        likelihood.load_state_dict(best_state["likelihood"])

    return model, likelihood


def walk_forward(symbol, window_train=180, step=5, horizon_test=5, on_window_complete=None):
    d = build_dataset(symbol)
    X, y, slices = assemble_tensor(d)

    records = []
    idx = window_train
    total_windows = max(1, (len(X) - window_train) // step + 1)
    window_count = 0
    while idx + horizon_test <= len(X):
        train_X_raw = X.iloc[idx - window_train : idx]
        test_X_raw = X.iloc[idx : idx + horizon_test]
        train_y = torch.tensor(y.iloc[idx - window_train : idx].values, dtype=torch.float32)
        test_y = torch.tensor(y.iloc[idx : idx + horizon_test].values, dtype=torch.float32)

        train_Xs, test_Xs = standardize(train_X_raw, test_X_raw)
        train_x = torch.tensor(train_Xs.values, dtype=torch.float32)
        test_x = torch.tensor(test_Xs.values, dtype=torch.float32)

        y_mean = train_y.mean()
        y_std = train_y.std()
        train_y_scaled = (train_y - y_mean) / y_std

        model, likelihood = train_gp(train_x, train_y_scaled, slices, verbose=(idx == window_train))
        model.eval()
        likelihood.eval()

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

        window_records = []
        for i in range(len(test_y)):
            window_records.append(
                {
                    "asset": symbol,
                    "date": test_X_raw.index[i],
                    "y_true": test_y[i].item(),
                    "pred_mean": float(np.asarray(pred_mean[i]).reshape(-1)[0]),
                    "pred_var": float(np.asarray(pred_var[i]).reshape(-1)[0]),
                    "var_fundamental": float(np.asarray(decomp["diagonal"]["fundamental"][i]).reshape(-1)[0]),
                    "var_technical": float(np.asarray(decomp["diagonal"]["technical"][i]).reshape(-1)[0]),
                    "var_sentiment": float(np.asarray(decomp["diagonal"]["sentiment"][i]).reshape(-1)[0]),
                    "var_interaction": float(np.asarray(decomp["interaction"][i]).reshape(-1)[0]),
                    "dominant_modality": dominant[i],
                }
            )

        records.extend(window_records)
        window_count += 1
        if on_window_complete is not None:
            on_window_complete(window_records, window_count, total_windows)

        idx += step

    df = pd.DataFrame.from_records(records)
    df.to_csv(os.path.join(OUTPUT_DIR, f"{symbol}_walk_forward_results.csv"), index=False)
    return df


def run_all():
    all_results = []
    for symbol in ASSETS:
        print(f"walk-forward: {symbol}")
        df = walk_forward(symbol)
        all_results.append(df)
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(os.path.join(OUTPUT_DIR, "all_assets_walk_forward_results.csv"), index=False)
    return combined


if __name__ == "__main__":
    run_all()
