"""Tests for the ECMWF-only daily post-processing direction."""

import unittest

import numpy as np

from s2s_models.daily_postprocessing import ECMWFDailyPostprocessor


class TestECMWFDailyPostprocessor(unittest.TestCase):
    def test_make_features_uses_only_ensemble_forecasts(self):
        rng = np.random.default_rng(42)
        members = rng.gamma(1.5, 4.0, size=(6, 10, 28))
        features = ECMWFDailyPostprocessor.make_features(members)

        self.assertEqual(features.shape, (6, 28, 6))
        np.testing.assert_allclose(features[:, :, 0], members.mean(axis=1))
        np.testing.assert_allclose(features[:, :, 3], np.median(members, axis=1))
        self.assertTrue(np.all((features[:, :, 5] >= 0.0) & (features[:, :, 5] <= 1.0)))

    def test_fit_predict_returns_nonnegative_daily_trajectory(self):
        rng = np.random.default_rng(7)
        members = rng.gamma(1.2, 3.0, size=(48, 10, 28))
        features = ECMWFDailyPostprocessor.make_features(members)
        targets = np.maximum(0.0, members.mean(axis=1) * 1.1 + rng.normal(0.0, 0.5, size=(48, 28)))

        model = ECMWFDailyPostprocessor(epochs=3, batch_size=16, hidden_dim=8)
        model.fit(features, targets, validation_data=(features[:8], targets[:8]))
        predictions = model.predict(features[:5])

        self.assertEqual(predictions.shape, (5, 28))
        self.assertTrue(np.all(np.isfinite(predictions)))
        self.assertTrue(np.all(predictions >= 0.0))
        self.assertEqual(len(model.history_["epoch"]), 3)
        self.assertIn("validation_rmse", model.history_)

    def test_predict_rejects_observation_history_shape(self):
        model = ECMWFDailyPostprocessor()
        with self.assertRaises(ValueError):
            model.predict(np.zeros((2, 7, 6)))

    def test_extreme_weight_is_recorded_in_training_history(self):
        rng = np.random.default_rng(11)
        members = rng.gamma(1.5, 4.0, size=(32, 10, 28))
        features = ECMWFDailyPostprocessor.make_features(members)
        targets = np.maximum(0.0, members.mean(axis=1) * 1.2 + rng.normal(0.0, 1.0, size=(32, 28)))

        model = ECMWFDailyPostprocessor(
            epochs=2,
            batch_size=8,
            hidden_dim=6,
            extreme_weight=0.75,
            extreme_threshold=5.0,
        )
        model.fit(features, targets, validation_data=(features[:6], targets[:6]))

        self.assertEqual(model.extreme_weight, 0.75)
        self.assertEqual(model.extreme_threshold, 5.0)
        self.assertIn("validation_extreme_loss", model.history_)


if __name__ == "__main__":
    unittest.main()
