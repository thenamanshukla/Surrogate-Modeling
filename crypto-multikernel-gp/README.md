# Multi-Kernel Additive Gaussian Process for Crypto Return Prediction with Per-Modality Uncertainty Decomposition

## Sandbox disclosure

This code was authored and reviewed in an environment with no internet access in the code-execution
sandbox and without `torch`, `gpytorch`, or `tensorflow` installed, so it could not be executed end to
end here against live multi-asset data. Every data source below is a real, working, keyless endpoint
(verified by direct fetch) except CoinGecko/CryptoCompare's OHLCV endpoints, which now require a free
API key or are robots-gated from automated fetchers — `src/data_acquisition.py` calls them the normal
way and will work from a machine with unrestricted internet. No synthetic data generation exists
anywhere in this codebase. Run `python -m src.data_acquisition` first; everything downstream consumes
only what lands in `data/`.

## Problem setup

For assets `{BTC, ETH, SOL, XRP, DOGE}`, predict the next-day log return `y_t = log(P_{t+1}/P_t)` from
three feature blocks observed at time `t`:

- **Fundamental** `x_f`: active addresses, tx count/value, NVT proxy (market cap / transfer volume),
  developer commit activity, circulating/max supply ratio, staking ratio where available.
- **Technical** `x_t`: lagged returns (1/3/7/14d), moving-average ratios, RSI-14, realized volatility
  (rolling std of returns, multiple windows), volume z-score.
- **Sentiment** `x_s`: FinBERT-scored news/social sentiment mean, sentiment volume (post count),
  sentiment momentum (Δ sentiment over 3d), and the market-wide Crypto Fear & Greed Index level + Δ.

## Model: Multi-Kernel Additive Deep GP

Each modality `m ∈ {f, t, s}` has its own small MLP encoder `φ_m: x_m → R^d` (deep kernel learning,
Wilson et al. 2016, "Deep Kernel Learning", AISTATS) feeding its own Matérn-5/2 kernel with a learned
output-scale:

```
k_m(x_m, x_m') = σ_m^2 · Matern52(φ_m(x_m), φ_m(x_m'))
k(x, x') = k_f(x_f, x_f') + k_t(x_t, x_t') + k_s(x_s, x_s')
```

This is exactly the additive-GP construction of Duvenaud, Nickisch & Rasmussen, "Additive Gaussian
Processes" (NeurIPS 2011): each `σ_m^2` is a first-order Sobol-style importance weight for modality `m`,
and because the kernel is a sum, `k(x,x)` decomposes exactly into per-modality diagonal terms.

Output-scales are optionally regime-gated: a small gating MLP takes a volatility/regime descriptor
`r_t` (e.g. 7-day realized vol z-score) and outputs a softmax reweighting of `[σ_f^2, σ_t^2, σ_s^2]`,
so the model can learn, e.g., that sentiment's relative weight rises in high-vol/news-driven regimes.

Likelihood: Student-t (`gpytorch.likelihoods.StudentTLikelihood`), fit with an `ApproximateGP` +
`VariationalELBO` (SVI) since the Student-t likelihood is non-conjugate — this follows the variational
inference recipe in Gardner et al., "GPyTorch: Blackbox Matrix-Matrix Gaussian Process Inference with
GPU Acceleration" (NeurIPS 2018).

## The novel evaluation: exact per-modality uncertainty decomposition

For an additive kernel, posterior predictive variance at `x*` is:

```
Var(x*) = k(x*,x*) - k(x*,X) K_inv k(X,x*)
```

Because `k(x*,X) = Σ_m k_m(x*,X)`, the correction term expands into a double sum over modality pairs.
We decompose exactly (no approximation) into a **diagonal term per modality** plus a **pairwise
interaction residual**:

```
D_m(x*)   = k_m(x*,x*) - k_m(x*,X) K_inv k_m(X,x*)          for m in {f, t, s}
C_{mn}(x*)= -2 · k_m(x*,X) K_inv k_n(X,x*)                   for m < n
Var(x*)   = D_f + D_t + D_s + C_ft + C_fs + C_ts             (exact identity)
```

`D_m` is reported as modality `m`'s attributed share of predictive uncertainty at each test point;
`ΣC_{mn}` is reported separately as a signed interaction term (negative = modalities are redundant at
that point, positive = they disagree / compound uncertainty). This is implemented in
`src/models/decomposition.py` and is the core novel contribution requested: nothing here approximates
or normalizes away the cross terms, so `D_f+D_t+D_s+ΣC` always equals the model's actual total variance
to machine precision — you can unit-test this identity directly.

## Evaluation protocol

- **Walk-forward** (expanding window, weekly refit) across all five assets, held out chronologically —
  no shuffling, no leakage.
- **Calibration**: empirical coverage of 50/80/90/95% predictive intervals, both pooled and *split by
  which modality's `D_m` was largest at that point* (does the model behave differently when sentiment
  dominates vs. when technicals dominate?).
- **Point-forecast metrics**: RMSE, MAE, directional accuracy.
- **Probabilistic metrics**: negative log predictive density (uses the Student-t likelihood directly),
  CRPS (Gneiting & Raftery, 2007) via Monte Carlo from the Student-t posterior.
- **Regime-shift stress test**: a labeled window around a known high-volatility news event per asset
  (populated from `data/regime_events.csv`, which you fill in from real dated events — e.g. an ETF
  approval, a de-peg, an exchange collapse) to check whether `D_s` (sentiment) spikes there and `D_f`
  (fundamental) stays flat, as hypothesized.
- **Baselines**: (1) Keras LSTM point-forecast + empirical residual-quantile intervals, (2) GARCH(1,1)
  volatility model (`arch` package) for the variance benchmark, (3) single-kernel (non-additive) GP.

## References

- Duvenaud, Nickisch, Rasmussen. "Additive Gaussian Processes." NeurIPS 2011.
- Wilson, Hu, Salakhutdinov, Xing. "Deep Kernel Learning." AISTATS 2016.
- Gardner, Pleiss, Bindel, Weinberger, Wilson. "GPyTorch." NeurIPS 2018. github.com/cornellius-gp/gpytorch
- Duvenaud. "Automatic Model Construction with Gaussian Processes." PhD thesis, Cambridge, 2014.
- Araci. "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models." arXiv:1908.10063, 2019.
  github.com/ProsusAI/finBERT
- Gneiting, Raftery. "Strictly Proper Scoring Rules, Prediction, and Estimation." JASA 2007. (CRPS)
- Rasmussen, Williams. "Gaussian Processes for Machine Learning." MIT Press, 2006.
- alternative.me Crypto Fear & Greed Index API: https://alternative.me/crypto/fear-and-greed-index/#api
- CoinGecko API docs: https://docs.coingecko.com/reference/coins-id-market-chart

## Layout

```
crypto_gp/
  src/
    data_acquisition.py     real API pulls -> data/*.csv, no synthetic fallback
    features.py              builds the three per-modality feature blocks + target
    models/
      encoders.py             per-modality MLP encoders
      multi_kernel_gp.py       additive deep-kernel GP, Student-t likelihood, regime gating
      decomposition.py         exact per-modality posterior-variance decomposition
      baselines.py             Keras LSTM + GARCH(1,1) + single-kernel GP baselines
    train.py                  walk-forward training loop across assets
    evaluate.py                calibration, CRPS, NLL, decomposition-vs-events plots
  data/
    regime_events.csv         you fill in real dated news/regime events per asset
  outputs/
```
