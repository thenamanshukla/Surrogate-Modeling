import unittest
import numpy as np
import pandas as pd
from scipy import stats

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluate import (
    point_metrics,
    coverage,
    crps_student_t,
    negative_log_predictive_density,
    fit_variance_recalibration_scale,
    apply_variance_recalibration,
)


class TestPointMetrics(unittest.TestCase):
    def test_perfect_predictions_zero_error(self):
        df = pd.DataFrame({"y_true": [0.01, -0.02, 0.03], "pred_mean": [0.01, -0.02, 0.03]})
        m = point_metrics(df)
        self.assertAlmostEqual(m["rmse"], 0.0, places=10)
        self.assertAlmostEqual(m["mae"], 0.0, places=10)
        self.assertAlmostEqual(m["directional_accuracy"], 1.0, places=10)

    def test_known_rmse_mae(self):
        df = pd.DataFrame({"y_true": [1.0, 2.0, 3.0], "pred_mean": [1.0, 2.0, 6.0]})
        m = point_metrics(df)
        self.assertAlmostEqual(m["rmse"], np.sqrt((0 + 0 + 9) / 3), places=10)
        self.assertAlmostEqual(m["mae"], (0 + 0 + 3) / 3, places=10)

    def test_directional_accuracy_all_wrong(self):
        df = pd.DataFrame({"y_true": [1.0, -1.0, 2.0], "pred_mean": [-1.0, 1.0, -2.0]})
        m = point_metrics(df)
        self.assertAlmostEqual(m["directional_accuracy"], 0.0, places=10)


class TestCoverageAgainstKnownAnalyticCase(unittest.TestCase):
    def test_coverage_matches_nominal_for_correctly_specified_model(self):
        rng = np.random.default_rng(42)
        dof = 4.0
        n = 20000
        true_scale = 0.02
        samples = true_scale * rng.standard_t(dof, size=n)
        df = pd.DataFrame({
            "y_true": samples,
            "pred_mean": np.zeros(n),
            "pred_var": np.full(n, true_scale ** 2 * dof / (dof - 2)),
        })
        for level in (0.5, 0.8, 0.9, 0.95):
            empirical = coverage(df, level, dof=dof)
            self.assertAlmostEqual(empirical, level, delta=0.02)

    def test_coverage_is_zero_when_interval_degenerate_and_miss(self):
        df = pd.DataFrame({"y_true": [1.0], "pred_mean": [0.0], "pred_var": [1e-12]})
        empirical = coverage(df, 0.5, dof=4.0)
        self.assertEqual(empirical, 0.0)

    def test_coverage_is_one_when_variance_huge(self):
        df = pd.DataFrame({"y_true": [0.5, -0.3, 0.1], "pred_mean": [0.0, 0.0, 0.0], "pred_var": [1e6, 1e6, 1e6]})
        empirical = coverage(df, 0.5, dof=4.0)
        self.assertEqual(empirical, 1.0)

    def test_coverage_increases_with_nominal_level(self):
        rng = np.random.default_rng(1)
        n = 500
        samples = 0.02 * rng.standard_t(4.0, size=n)
        df = pd.DataFrame({
            "y_true": samples,
            "pred_mean": np.zeros(n),
            "pred_var": np.full(n, 0.02 ** 2 * 4.0 / (4.0 - 2)),
        })
        levels = [0.5, 0.8, 0.9, 0.95]
        empirical = [coverage(df, lv, dof=4.0) for lv in levels]
        self.assertTrue(all(empirical[i] <= empirical[i + 1] for i in range(len(empirical) - 1)))


class TestNegativeLogPredictiveDensity(unittest.TestCase):
    def test_matches_scipy_directly(self):
        y_true = np.array([0.1, -0.2, 0.05])
        mean = np.array([0.0, 0.0, 0.0])
        var = np.array([0.01, 0.01, 0.01])
        dof = 4.0
        got = negative_log_predictive_density(y_true, mean, var, dof=dof)
        scale = np.sqrt(var * (dof - 2) / dof)
        expected = -np.mean(stats.t.logpdf(y_true, df=dof, loc=mean, scale=scale))
        self.assertAlmostEqual(got, expected, places=10)

    def test_lower_nll_for_correct_vs_overconfident_model(self):
        rng = np.random.default_rng(7)
        dof = 4.0
        true_scale = 0.03
        y_true = true_scale * rng.standard_t(dof, size=2000)
        mean = np.zeros_like(y_true)
        correct_var = np.full_like(y_true, true_scale ** 2 * dof / (dof - 2))
        overconfident_var = correct_var * 0.01
        nll_correct = negative_log_predictive_density(y_true, mean, correct_var, dof=dof)
        nll_overconfident = negative_log_predictive_density(y_true, mean, overconfident_var, dof=dof)
        self.assertLess(nll_correct, nll_overconfident)


class TestCRPS(unittest.TestCase):
    def test_crps_non_negative(self):
        rng = np.random.default_rng(3)
        y_true = rng.normal(0, 0.02, size=50)
        mean = np.zeros(50)
        var = np.full(50, 0.02 ** 2)
        vals = crps_student_t(y_true, mean, var, dof=6.0)
        self.assertTrue((vals >= -1e-8).all())

    def test_crps_zero_variance_reduces_to_absolute_error(self):
        y_true = np.array([0.05, -0.03])
        mean = np.array([0.0, 0.0])
        var = np.array([1e-10, 1e-10])
        vals = crps_student_t(y_true, mean, var, dof=6.0, n_samples=500)
        expected = np.abs(y_true - mean)
        np.testing.assert_allclose(vals, expected, atol=1e-3)


class TestVarianceRecalibration(unittest.TestCase):
    def test_scale_of_one_when_already_calibrated(self):
        rng = np.random.default_rng(9)
        dof = 4.0
        true_var = 0.001
        n = 5000
        errors = np.sqrt(true_var) * rng.standard_normal(n)
        df = pd.DataFrame({
            "y_true": errors,
            "pred_mean": np.zeros(n),
            "pred_var": np.full(n, true_var * dof / (dof - 2)),
        })
        scale = fit_variance_recalibration_scale(df, dof=dof)
        self.assertAlmostEqual(scale, 1.0, delta=0.15)

    def test_recalibration_preserves_relative_modality_shares(self):
        df = pd.DataFrame({
            "y_true": [0.01, 0.02, -0.01],
            "pred_mean": [0.0, 0.0, 0.0],
            "pred_var": [0.001, 0.002, 0.0015],
            "var_fundamental": [0.0003, 0.0004, 0.0005],
            "var_technical": [0.0004, 0.0009, 0.0006],
            "var_sentiment": [0.0003, 0.0007, 0.0004],
            "var_interaction": [0.0, 0.0, 0.0],
        })
        calibrated = apply_variance_recalibration(df, scale=2.0)
        original_share = df["var_fundamental"] / df["pred_var"]
        calibrated_share = calibrated["var_fundamental"] / calibrated["pred_var"]
        np.testing.assert_allclose(original_share, calibrated_share, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
