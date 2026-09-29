"""Unit tests for Adaptive Bias Correction (ABC) and Gated-ABC models."""

import unittest
import numpy as np
import pandas as pd

from s2s_models.abc import (
    ClimatologyPlusPlus,
    DynamicalPlusPlus,
    GatedABC,
    OpportunisticABC,
    PersistencePlusPlus,
    StandardABC,
)
from s2s_models.metrics import (
    anomaly_correlation_coefficient,
    mean_absolute_error,
    root_mean_squared_error,
)


class TestABCModels(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.n = 200
        self.lat = np.random.choice([30.0, 35.0, 40.0], size=self.n)
        self.lon = np.random.choice([250.0, 260.0, 270.0], size=self.n)
        self.ens_mean = np.random.uniform(5.0, 40.0, size=self.n)
        self.ens_std = np.random.uniform(1.0, 15.0, size=self.n)
        self.tau = np.random.uniform(50.0, 100.0, size=self.n)
        self.y_precip = np.random.uniform(0.0, 35.0, size=self.n)
        self.y_extreme = (self.y_precip > self.tau * 0.8).astype(int)

        self.doy = np.random.randint(1, 366, size=self.n)
        self.doy_sin = np.sin(2 * np.pi * self.doy / 365.25)
        self.doy_cos = np.cos(2 * np.pi * self.doy / 365.25)

        self.df = pd.DataFrame({
            "ens_mean": self.ens_mean,
            "ens_std": self.ens_std,
            "threshold": self.tau,
            "mean_to_threshold": self.ens_mean / self.tau,
            "spread_to_threshold": self.ens_std / self.tau,
            "lat": self.lat,
            "lon": self.lon,
            "doy_sin": self.doy_sin,
            "doy_cos": self.doy_cos,
            "oni": np.random.uniform(-1.5, 1.5, size=self.n),
            "mjo_amplitude": np.random.uniform(0.2, 2.0, size=self.n),
            "mjo_phase": np.random.choice([1, 2, 3, 4, 5, 6, 7, 8], size=self.n).astype(float),
        })

    def test_dynamical_plus_plus(self):
        dyn = DynamicalPlusPlus(modulate_by_spread=True)
        dyn.fit(self.ens_mean, self.y_precip, self.lat, self.lon)
        clim_ref = np.full(self.n, 15.0)
        preds = dyn.predict(self.ens_mean, self.lat, self.lon, ens_std=self.ens_std, clim_ref=clim_ref)
        self.assertEqual(len(preds), self.n)
        self.assertTrue(np.all(preds >= 0.0))

    def test_climatology_plus_plus(self):
        clim = ClimatologyPlusPlus()
        clim.fit(self.y_precip, self.lat, self.lon, self.doy)
        preds = clim.predict(self.lat, self.lon, self.doy)
        self.assertEqual(len(preds), self.n)
        self.assertTrue(np.all(preds >= 0.0))

    def test_persistence_plus_plus(self):
        pers = PersistencePlusPlus()
        lagged_obs = self.y_precip * 0.9
        clim_preds = np.full(self.n, 12.0)
        pers.fit(self.y_precip, self.ens_mean, clim_preds, lagged_obs)
        preds = pers.predict(self.ens_mean, clim_preds, lagged_obs)
        self.assertEqual(len(preds), self.n)
        self.assertTrue(np.all(preds >= 0.0))

    def test_standard_abc(self):
        model = StandardABC(lead_week=2)
        model.fit(self.df, self.y_precip)
        preds = model.predict(self.df)
        self.assertEqual(len(preds), self.n)
        self.assertTrue(np.all(preds >= 0.0))

        y_dyn, y_clim, y_pers = model.predict_components(self.df)
        expected = (1.0 / 3.0) * y_dyn + (1.0 / 3.0) * y_clim + (1.0 / 3.0) * y_pers
        np.testing.assert_allclose(preds, expected, atol=1e-5)

        p_ext = model.predict_extreme_proba(self.df, self.tau)
        self.assertEqual(len(p_ext), self.n)
        self.assertTrue(np.all((p_ext >= 0.0) & (p_ext <= 1.0)))

    def test_opportunistic_abc(self):
        model = OpportunisticABC(lead_week=2)
        model.fit(self.df, self.y_precip)
        preds = model.predict(self.df)
        self.assertEqual(len(preds), self.n)
        self.assertTrue(np.all(preds >= 0.0))

    def test_gated_abc(self):
        model = GatedABC(lead_week=2, hidden_dim=16, epochs=5, batch_size=32)
        model.fit(self.df, self.y_precip, self.y_extreme)
        preds, weights = model.predict_with_weights(self.df)

        self.assertEqual(len(preds), self.n)
        self.assertEqual(weights.shape, (self.n, 3))
        # Verify softmax weights sum to 1.0
        np.testing.assert_allclose(np.sum(weights, axis=-1), np.ones(self.n), atol=1e-4)
        self.assertTrue(np.all(weights >= 0.0) and np.all(weights <= 1.0))
        self.assertTrue(np.all(preds >= 0.0))

        p_ext = model.predict_extreme_proba(self.df, self.tau)
        self.assertTrue(np.all((p_ext >= 0.0) & (p_ext <= 1.0)))

    def test_continuous_metrics(self):
        y_true = np.array([10.0, 20.0, 30.0, 40.0])
        y_pred = np.array([12.0, 18.0, 33.0, 38.0])
        y_clim = np.array([15.0, 15.0, 15.0, 15.0])

        rmse = root_mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        acc = anomaly_correlation_coefficient(y_true, y_pred, y_clim)

        self.assertGreater(rmse, 0.0)
        self.assertGreater(mae, 0.0)
        self.assertGreater(acc, 0.8)  # highly correlated predictions


if __name__ == "__main__":
    unittest.main()

