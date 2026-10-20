"""Tests for ECMWF-only daily benchmark post-processors."""

import unittest

import numpy as np

from s2s_models.daily_benchmarks import (
    DeepSetDailyPostprocessor,
    EmpiricalQuantileMapping,
    LeadBiasCorrection,
    LeadClimatology,
)


class TestDailyBenchmarks(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(21)
        self.members = rng.gamma(1.5, 3.0, size=(20, 10, 28)).astype(float)
        self.baseline = self.members.mean(axis=1)
        self.targets = np.maximum(0.0, self.baseline * 1.1 + rng.normal(0.0, 0.5, size=(20, 28)))

    def test_statistical_benchmarks_return_nonnegative_trajectories(self):
        climatology = LeadClimatology().fit(self.targets)
        bias = LeadBiasCorrection().fit(self.baseline, self.targets)
        mapping = EmpiricalQuantileMapping().fit(self.baseline, self.targets)

        for prediction in (
            climatology.predict(4),
            bias.predict(self.baseline[:4]),
            mapping.predict(self.baseline[:4]),
        ):
            self.assertEqual(prediction.shape, (4, 28))
            self.assertTrue(np.all(np.isfinite(prediction)))
            self.assertTrue(np.all(prediction >= 0.0))

    def test_deepset_benchmark_is_permutation_invariant(self):
        model = DeepSetDailyPostprocessor(epochs=2, hidden_dim=6, batch_size=8)
        model.fit(self.members, self.targets, validation_data=(self.members[:4], self.targets[:4]))
        first = model.predict(self.members[:3])
        permuted = model.predict(self.members[:3, ::-1])

        self.assertEqual(first.shape, (3, 28))
        self.assertTrue(np.all(first >= 0.0))
        np.testing.assert_allclose(first, permuted, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()