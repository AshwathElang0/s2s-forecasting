"""A 3D U-Net that corrects a bias-corrected forecast instead of repainting it.

    prediction = anchor + U-Net(inputs)          (ResidualUNet3D, log1p scale)
    prediction = anchor x exp(U-Net(inputs))     (MultiplicativeResidualUNet3D, mm scale)

`anchor` is the Dyn++ forecast (preprocess/bias_correction.py) on the same
scale as the target. The head is zero-initialised, so before training the
model *is* Dyn++, and training only has to learn what Dyn++ gets wrong. The
trunk is the unchanged UNet3D, as in model/gated_unet.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from model.unet3d import UNet3D


class ResidualDataset(Dataset):
    """(features, target, anchor, climatology) per sample, all (C or 1, lead, H, W)."""

    def __init__(self, imgs, labels, anchors, clims):
        self.imgs, self.labels, self.anchors, self.clims = imgs, labels, anchors, clims

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "imgs": torch.as_tensor(self.imgs[idx], dtype=torch.float32),
            "labels": torch.as_tensor(self.labels[idx], dtype=torch.float32),
            "anchor": torch.as_tensor(self.anchors[idx], dtype=torch.float32),
            "clim": torch.as_tensor(self.clims[idx], dtype=torch.float32),
        }


class ResidualUNet3D(UNet3D):
    def __init__(self, in_channels: int, unit: int = 64):
        super().__init__(in_channels, unit)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        return anchor + super().forward(x)


class MultiplicativeResidualUNet3D(ResidualUNet3D):
    """prediction = anchor x exp(U-Net(inputs)), for an anchor in mm/day (scaled).

    Stays non-negative and starts at exactly the anchor (exp(0) = 1), like
    Dyn++ itself, which is a multiplicative correction. The log-factor is
    clamped to [-4, 4] (x0.02 to x55) so one bad step cannot overflow."""

    def forward(self, x: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        log_factor = UNet3D.forward(self, x).float().clamp(-4.0, 4.0)
        return anchor * torch.exp(log_factor)


class PlainUNet3D(UNet3D):
    """The headline U-Net with the residual model's call signature; ignores the anchor."""

    def forward(self, x: torch.Tensor, anchor: torch.Tensor | None = None) -> torch.Tensor:
        return super().forward(x)


def area_mean(x: torch.Tensor, lat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """(B, 1, lead, H, W) -> (B, lead) latitude-weighted mean over valid cells."""
    w = torch.cos(torch.deg2rad(lat)).to(x.dtype).view(-1, 1) * mask.to(x.dtype)
    return (x.squeeze(1) * w).sum(dim=(-2, -1)) / w.sum()
