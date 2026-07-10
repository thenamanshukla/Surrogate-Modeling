import os
import json
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _rsi(returns, window=14):
    gains = returns.clip(lower=0)
    losses = -returns.clip(upper=0)
    avg_gain = gains.rolling(window).mean()
    avg_loss = losses.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def build_technical_features(chart):
    df = pd.DataFrame(index=chart.index)
    ret = np.log(chart["price"]).diff()
    df["ret_lag1"] = ret.shift(1)
    df["ret_lag3"] = ret.rolling(3).sum().shift(1)
    df["ret_lag7"] = ret.rolling(7).sum().shift(1)
    df["ret_lag14"] = ret.rolling(14).sum().shift(1)
    df["ma_ratio_7_30"] = (chart["price"].rolling(7).mean() / chart["price"].rolling(30).mean()).shift(1)
    df["rsi_14"] = _rsi(ret, 14).shift(1)
    df["realized_vol_7"] = ret.rolling(7).std().shift(1)
    df["realized_vol_30"] = ret.rolling(30).std().shift(1)
    vol_mean = chart["volume"].rolling(30).mean()
    vol_std = chart["volume"].rolling(30).std()
    df["volume_zscore"] = ((chart["volume"] - vol_mean) / vol_std).shift(1)
    return df


def build_fundamental_features(chart, snapshot):
    df = pd.DataFrame(index=chart.index)
    df["nvt_proxy"] = (chart["market_cap"] / chart["volume"].rolling(7).mean()).shift(1)
    df["market_cap_change_30"] = chart["market_cap"].pct_change(30).shift(1)
    circ = snapshot.get("circulating_supply")
    total = snapshot.get("total_supply")
    maxs = snapshot.get("max_supply")
    if circ and maxs:
        df["supply_ratio"] = circ / maxs
    elif circ and total:
        df["supply_ratio"] = circ / total
    else:
        df["supply_ratio"] = 1.0
    df["dev_commits_4w"] = snapshot.get("commit_count_4_weeks")
    df["dev_prs_merged"] = snapshot.get("pull_requests_merged")
    df["reddit_subscribers"] = snapshot.get("reddit_subscribers")
    df = df.fillna(0).infer_objects()
    df = df.ffill()
    return df


def build_sentiment_features(chart, fng):
    df = pd.DataFrame(index=chart.index)
    fng_aligned = fng.reindex(chart.index).ffill()
    df["fear_greed"] = fng_aligned["fear_greed"].shift(1)
    df["fear_greed_delta_3"] = fng_aligned["fear_greed"].diff(3).shift(1)
    df["fear_greed_delta_1"] = fng_aligned["fear_greed"].diff(1).shift(1)
    df["fear_greed_zscore_30"] = (
        (fng_aligned["fear_greed"] - fng_aligned["fear_greed"].rolling(30).mean())
        / fng_aligned["fear_greed"].rolling(30).std()
    ).shift(1)
    return df


def build_regime_descriptor(chart):
    ret = np.log(chart["price"]).diff()
    vol7 = ret.rolling(7).std()
    z = (vol7 - vol7.rolling(90).mean()) / vol7.rolling(90).std()
    return z.shift(1).rename("regime_vol_zscore")


def build_target(chart):
    return np.log(chart["price"]).diff().shift(-1).rename("target_next_return")


def build_dataset(symbol):
    chart = pd.read_csv(
        os.path.join(DATA_DIR, f"{symbol}_market_chart.csv"), index_col=0, parse_dates=True
    )
    with open(os.path.join(DATA_DIR, f"{symbol}_snapshot.json")) as fh:
        snapshot = json.load(fh)
    fng = pd.read_csv(os.path.join(DATA_DIR, "fear_greed_index.csv"), index_col=0, parse_dates=True)

    technical = build_technical_features(chart)
    fundamental = build_fundamental_features(chart, snapshot)
    sentiment = build_sentiment_features(chart, fng)
    regime = build_regime_descriptor(chart)
    target = build_target(chart)

    full = pd.concat([technical, fundamental, sentiment, regime, target], axis=1)
    full = full.dropna()
    return {
        "technical": full[technical.columns],
        "fundamental": full[fundamental.columns],
        "sentiment": full[sentiment.columns],
        "regime": full[["regime_vol_zscore"]],
        "target": full["target_next_return"],
    }


if __name__ == "__main__":
    for sym in ["BTC", "ETH", "SOL", "XRP", "DOGE"]:
        d = build_dataset(sym)
        print(sym, d["target"].shape, d["technical"].shape, d["fundamental"].shape, d["sentiment"].shape)
