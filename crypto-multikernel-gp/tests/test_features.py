import unittest
import numpy as np
import pandas as pd

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import (
    build_technical_features,
    build_fundamental_features,
    build_sentiment_features,
    build_regime_descriptor,
    build_target,
)


def make_synthetic_chart(n=60, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    price = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, size=n)))
    volume = rng.uniform(1e6, 5e6, size=n)
    market_cap = price * 1e6
    return pd.DataFrame({"price": price, "volume": volume, "market_cap": market_cap}, index=dates)


class TestTechnicalFeatures(unittest.TestCase):
    def setUp(self):
        self.chart = make_synthetic_chart()
        self.tech = build_technical_features(self.chart)

    def test_no_lookahead_ret_lag1(self):
        actual_ret = np.log(self.chart["price"]).diff()
        for i in range(2, len(self.chart)):
            expected = actual_ret.iloc[i - 1]
            got = self.tech["ret_lag1"].iloc[i]
            if not np.isnan(expected):
                self.assertAlmostEqual(got, expected, places=10)

    def test_columns_present(self):
        expected_cols = {
            "ret_lag1", "ret_lag3", "ret_lag7", "ret_lag14",
            "ma_ratio_7_30", "rsi_14", "realized_vol_7",
            "realized_vol_30", "volume_zscore",
        }
        self.assertTrue(expected_cols.issubset(set(self.tech.columns)))

    def test_rsi_bounded(self):
        valid_rsi = self.tech["rsi_14"].dropna()
        self.assertTrue((valid_rsi >= 0).all())
        self.assertTrue((valid_rsi <= 100).all())

    def test_same_index_as_chart(self):
        self.assertTrue(self.tech.index.equals(self.chart.index))


class TestFundamentalFeatures(unittest.TestCase):
    def test_supply_ratio_uses_max_supply_when_available(self):
        chart = make_synthetic_chart()
        snapshot = {"circulating_supply": 19_000_000, "total_supply": 21_000_000, "max_supply": 21_000_000}
        fund = build_fundamental_features(chart, snapshot)
        self.assertAlmostEqual(fund["supply_ratio"].iloc[-1], 19_000_000 / 21_000_000, places=8)

    def test_supply_ratio_falls_back_to_total_supply(self):
        chart = make_synthetic_chart()
        snapshot = {"circulating_supply": 120_000_000, "total_supply": 120_000_000, "max_supply": None}
        fund = build_fundamental_features(chart, snapshot)
        self.assertAlmostEqual(fund["supply_ratio"].iloc[-1], 1.0, places=8)

    def test_supply_ratio_neutral_default_when_nothing_available(self):
        chart = make_synthetic_chart()
        snapshot = {"circulating_supply": None, "total_supply": None, "max_supply": None}
        fund = build_fundamental_features(chart, snapshot)
        self.assertTrue((fund["supply_ratio"] == 1.0).all())

    def test_no_nans_after_fill(self):
        chart = make_synthetic_chart()
        snapshot = {"circulating_supply": None, "total_supply": None, "max_supply": None,
                    "commit_count_4_weeks": None, "pull_requests_merged": None, "reddit_subscribers": None}
        fund = build_fundamental_features(chart, snapshot)
        self.assertFalse(fund.isna().any().any())


class TestSentimentFeatures(unittest.TestCase):
    def test_fear_greed_shift_no_lookahead(self):
        chart = make_synthetic_chart()
        fng = pd.DataFrame(
            {"fear_greed": np.arange(len(chart))},
            index=chart.index,
        )
        sent = build_sentiment_features(chart, fng)
        for i in range(1, len(chart)):
            self.assertAlmostEqual(sent["fear_greed"].iloc[i], fng["fear_greed"].iloc[i - 1], places=8)


class TestTargetConstruction(unittest.TestCase):
    def test_target_is_next_day_log_return(self):
        chart = make_synthetic_chart()
        target = build_target(chart)
        log_ret = np.log(chart["price"]).diff()
        for i in range(len(chart) - 1):
            expected = log_ret.iloc[i + 1]
            got = target.iloc[i]
            if not np.isnan(expected):
                self.assertAlmostEqual(got, expected, places=10)

    def test_last_row_is_nan(self):
        chart = make_synthetic_chart()
        target = build_target(chart)
        self.assertTrue(np.isnan(target.iloc[-1]))


class TestRegimeDescriptor(unittest.TestCase):
    def _make_transition_chart(self):
        rng = np.random.default_rng(1)
        n_calm, n_vol = 150, 100
        n = n_calm + n_vol
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        calm_returns = rng.normal(0, 0.005, size=n_calm)
        volatile_returns = rng.normal(0, 0.05, size=n_vol)
        rets = np.concatenate([calm_returns, volatile_returns])
        price = 100 * np.exp(np.cumsum(rets))
        chart = pd.DataFrame({"price": price}, index=dates)
        return chart, n_calm

    def test_regime_zscore_spikes_at_volatility_transition(self):
        chart, transition_idx = self._make_transition_chart()
        regime = build_regime_descriptor(chart)
        spike_window = regime.iloc[transition_idx : transition_idx + 10]
        deep_calm_baseline = regime.iloc[transition_idx - 30 : transition_idx - 10].mean()
        self.assertFalse(spike_window.isna().any())
        self.assertFalse(np.isnan(deep_calm_baseline))
        self.assertGreater(spike_window.mean(), deep_calm_baseline + 1.0)

    def test_regime_zscore_mean_reverts_after_baseline_adapts(self):
        chart, transition_idx = self._make_transition_chart()
        regime = build_regime_descriptor(chart)
        spike_peak = regime.iloc[transition_idx : transition_idx + 10].mean()
        late_adapted = regime.iloc[-10:].mean()
        self.assertGreater(spike_peak, late_adapted)


if __name__ == "__main__":
    unittest.main()
