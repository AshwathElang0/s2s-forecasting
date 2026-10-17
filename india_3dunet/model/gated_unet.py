"""The gated residual head (experiments/exp03_departure_from_normal/DESIGN.md S2.2).

Subclasses the existing model/unet3d.py rather than forking it: the encoder,
decoder, skip connections, pooling and loss functions are all the unmodified
originals, and only the final projection changes. `UNet3D.forward` ends with
`return self.head(out)`, so swapping `self.head` for a two-channel conv and
calling `super().forward()` reuses the whole trunk with no duplication.

NB: this file must not be named `model.py` -- a `model.py` next to a script
would land on sys.path[0] and shadow the top-level `model/` package this very
module imports from.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from model.unet3d import UNet3D


class GatedDataset(Dataset):
    """(features, target, anchor) triples."""

    def __init__(self, imgs, labels, anchors):
        self.imgs = imgs
        self.labels = labels
        self.anchors = anchors

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "imgs": torch.as_tensor(self.imgs[idx], dtype=torch.float32),
            "labels": torch.as_tensor(self.labels[idx], dtype=torch.float32),
            "anchor": torch.as_tensor(self.anchors[idx], dtype=torch.float32),
        }


class GatedUNet3D(UNet3D):
    """Emits `gate * anchor + correction` instead of the field itself.

    `gate` is a per-cell, per-lead-day sigmoid, so the network can trust ECMWF
    at day 2 over the Ghats and ignore it at day 30 over the Deccan
    independently. With both anomalies scaled to unit variance the optimal
    coefficient on the anchor is the correlation r, which lives in [0, 1] --
    so the sigmoid spans exactly the useful range, and as skill decays with
    lead, r -> 0 drives gate -> 0 and the forecast falls back to climatology
    on its own.
    """

    def __init__(self, in_channels: int, unit: int = 64, gate_bias_init: float = 0.0):
        super().__init__(in_channels, unit)
        self.head = nn.Conv3d(unit, 2, kernel_size=3, stride=1, padding=1)

        # Start at `sigmoid(gate_bias_init) x anchor` with no learned
        # correction, so epoch 0 is already a real forecast rather than noise.
        # Zeroing the final projection is standard: weight gradients stay
        # non-zero, only the first step's gradient to the encoder vanishes.
        #
        # Default 0.0 (gate = 0.5) rather than something confident: with both
        # anomalies on unit variance the loss-optimal gate is the correlation
        # r, measured on the train split at 0.53 (day 1) decaying to 0.04
        # (day 32), mean ~0.25. A gate of 0.88 would start *worse* than plain
        # climatology; 0.5 ties climatology on average and beats it at short
        # lead, which is the intended starting point.
        nn.init.zeros_(self.head.weight)
        with torch.no_grad():
            self.head.bias.copy_(torch.tensor([gate_bias_init, 0.0]))

    def forward(self, x: torch.Tensor, anchor: torch.Tensor, return_gate: bool = False):
        h = super().forward(x)              # (B, 2, lead, H, W) -- head was swapped
        gate = torch.sigmoid(h[:, 0:1])
        pred = gate * anchor + h[:, 1:2]
        return (pred, gate) if return_gate else pred


class PlainAnomalyUNet3D(UNet3D):
    """The original single-channel head, predicting the departure from normal
    directly. No gate, no anchor blend: the control that separates the effect
    of the anomaly target from the effect of the gate."""

    def forward(self, x: torch.Tensor, anchor: torch.Tensor | None = None, return_gate: bool = False):
        pred = super().forward(x)
        return (pred, None) if return_gate else pred
