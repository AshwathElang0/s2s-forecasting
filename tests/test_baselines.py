"""Unit tests for S2S prediction baselines and metrics."""

import unittest
import numpy as np
import pandas as pd

from s2s_models.baselines import (
    ClimatologyBaseline,
    FixedCorrectionBaseline,
    RawEnsembleBaseline,
    SpreadGatedBaseline,
)
from s2s_models.metrics import (
    brier_score,
    brier_skill_score,
    compute_all_metrics,
    reliability_curve_data,
)


class TestMetrics(unittest.TestCase):
    def test_perfect_forecast(self):
        y = np.array([0, 1, 0, 1])
        p = np.array([0.0, 1.0, 0.0, 1.0])
        self.assertAlmostEqual(brier_score(y, p), 0.0)
        self.assertAlmostEqual(brier_skill_score(y, p), 1.0)

    def test_climatology_bss(self):
        y = np.array([0, 1, 0, 0])
        p_clim = np.full_like(y, 0.25, dtype=float)
        # BSS of climatology against itself should be 0.0
        self.assertAlmostEqual(brier_skill_score(y, p_clim, y_ref_prob=p_clim), 0.0)

    def test_worse_than_climatology(self):
        y = np.array([0, 0, 0, 0])
        p_bad = np.array([1.0, 1.0, 1.0, 1.0])
        bss = brier_skill_score(y, p_bad, y_ref_prob=np.array([0.01, 0.01, 0.01, 0.01]))
        self.assertLess(bss, 0.0)

    def test_compute_all_metrics(self):
        y = np.array([0, 0, 1, 1, 0, 0, 0, 1])
        p = np.array([0.1, 0.2, 0.8, 0.9, 0.2, 0.1, 0.3, 0.7])
        m = compute_all_metrics(y, p)
        self.assertIn("brier_score", m)
        self.assertIn("brier_skill_score", m)
        self.assertIn("roc_auc", m)
        self.assertGreater(m["roc_auc"], 0.8)

    def test_reliability_curve(self):
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        p = np.array([0.05, 0.08, 0.1, 0.12, 0.85, 0.88, 0.9, 0.95])
        cal = reliability_curve_data(y, p, n_bins=5)
        self.assertEqual(len(cal["bin_centers"]), 5)
        self.assertEqual(np.sum(cal["bin_counts"]), 8)


class TestBaselines(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.n = 200
        self.y = (np.random.rand(self.n) < 0.05).astype(int)
        self.p_raw = np.random.choice([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], size=self.n)
        self.spread = np.random.uniform(5.0, 50.0, size=self.n)
        self.X = pd.DataFrame({
            "ens_mean": np.random.uniform(0, 50, self.n),
            "ens_std": self.spread,
            "raw_exceed_frac": self.p_raw,
            "threshold": np.full(self.n, 60.0),
            "lat": np.random.uniform(25, 45, self.n),
            "lon": np.random.uniform(240, 290, self.n),
        })

    def test_raw_ensemble(self):
        m = RawEnsembleBaseline().fit(self.X, self.y)
        preds = m.predict_proba(self.p_raw)
        np.testing.assert_array_equal(preds, self.p_raw)

    def test_climatology_baseline(self):
        m = ClimatologyBaseline().fit(self.X, self.y)
        preds = m.predict_proba(self.n)
        self.assertAlmostEqual(preds[0], np.mean(self.y))

    def test_fixed_correction(self):
        m = FixedCorrectionBaseline(model_type="hist_gb", max_iter=10)
        m.fit(self.X, self.y)
        preds = m.predict_proba(self.X)
        self.assertEqual(len(preds), self.n)
        self.assertTrue(np.all((preds >= 0.0) & (preds <= 1.0)))

    def test_spread_gated_behavior(self):
        base = FixedCorrectionBaseline(model_type="hist_gb", max_iter=10)
        gated = SpreadGatedBaseline(base_model=base, gate_method="percentile")
        gated.fit(self.X, self.y, self.spread)

        # High spread should have low confidence; low spread should have high confidence
        low_spread = np.array([1.0])
        high_spread = np.array([100.0])
        c_low = gated.compute_confidence(low_spread)[0]
        c_high = gated.compute_confidence(high_spread)[0]
        self.assertGreater(c_low, c_high)

        preds = gated.predict_proba(self.X, self.spread)
        self.assertEqual(len(preds), self.n)
        self.assertTrue(np.all((preds >= 0.0) & (preds <= 1.0)))


if __name__ == "__main__":
    unittest.main()

