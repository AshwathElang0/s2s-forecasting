"""Unit tests for PyTorch-based Joint Confidence-Gated Neural Network."""

import unittest
import numpy as np
import pandas as pd
import torch

from s2s_models.neural_gated import JointConfidenceGatedNet, NeuralGatedBaseline


class TestNeuralGated(unittest.TestCase):
    def test_forward_pass_and_shapes(self):
        batch_size = 16
        in_features = 10
        net = JointConfidenceGatedNet(in_features=in_features, p_clim=0.015, hidden_dim=32)
        x = torch.randn(batch_size, in_features)

        p_gated, p_event, conf = net(x)
        self.assertEqual(p_gated.shape, (batch_size,))
        self.assertEqual(p_event.shape, (batch_size,))
        self.assertEqual(conf.shape, (batch_size,))

        # Verify all probabilities and confidences are bounded in [0, 1]
        self.assertTrue(torch.all(p_gated >= 0.0) and torch.all(p_gated <= 1.0))
        self.assertTrue(torch.all(p_event >= 0.0) and torch.all(p_event <= 1.0))
        self.assertTrue(torch.all(conf >= 0.0) and torch.all(conf <= 1.0))

    def test_gradient_flow_to_both_heads(self):
        """Verify that gradients from the Brier loss flow to both the event and confidence heads."""
        net = JointConfidenceGatedNet(in_features=8, p_clim=0.01, hidden_dim=16)
        x = torch.randn(10, 8)
        y = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])

        p_gated, _, _ = net(x)
        loss = torch.nn.MSELoss()(p_gated, y)
        loss.backward()

        # Check gradients in event head
        event_grad = net.event_head[-1].weight.grad
        self.assertIsNotNone(event_grad)
        self.assertFalse(torch.all(event_grad == 0))

        # Check gradients in confidence head (the learnable weighting factor)
        conf_grad = net.confidence_head[-1].weight.grad
        self.assertIsNotNone(conf_grad)
        self.assertFalse(torch.all(conf_grad == 0))

    def test_sklearn_wrapper_fit_predict(self):
        np.random.seed(42)
        n = 300
        n_feat = 6
        X = pd.DataFrame(np.random.randn(n, n_feat), columns=[f"f_{i}" for i in range(n_feat)])
        y = (np.random.rand(n) < 0.05).astype(int)

        model = NeuralGatedBaseline(hidden_dim=16, epochs=5, batch_size=64)
        model.fit(X, y)

        preds, conf = model.predict_proba(X, return_confidence=True)
        self.assertEqual(len(preds), n)
        self.assertEqual(len(conf), n)
        self.assertTrue(np.all((preds >= 0.0) & (preds <= 1.0)))
        self.assertTrue(np.all((conf >= 0.0) & (conf <= 1.0)))


if __name__ == "__main__":
    unittest.main()

