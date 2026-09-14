"""3D U-Net for ECMWF S2S post-processing over India.

Architecture is a direct port of the reference paper's `model/IRnAF.py`
(github.com/W-US-BCSR/IRnAF) -- same 3-stage encoder/decoder, same
Conv3d(k=3)+BatchNorm3d+GELU blocks, same average-pool downsampling and
transposed-conv upsampling. See plan.md S1.1/S2 for what changed and why:

- single continuous output channel (already true in the reference too)
- masked MSE and masked, latitude-weighted pattern-correlation loss,
  because our domain has ocean/foreign-territory cells IMD has no data
  for (the reference's CONUS-west box didn't need this)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """(features, target) pairs, matching the reference's CustomDataset."""

    def __init__(self, imgs, labels):
        self.imgs = imgs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "imgs": torch.as_tensor(self.imgs[idx], dtype=torch.float32),
            "labels": torch.as_tensor(self.labels[idx], dtype=torch.float32),
        }


def _cbr3d(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_ch),
        nn.GELU(),
    )


class UNet3D(nn.Module):
    """Input: (batch, channels, lead, lat, lon). Output: (batch, 1, lead, lat, lon).

    `channels` folds together whatever the caller has already stacked into
    the channel axis -- ensemble members, variables, or engineered features
    (mean/spread/orography/...). Unlike the reference implementation we do
    *not* reshape (ens, var) into channels inside the model; the caller is
    responsible for handing us a single channel axis, since our feature set
    (mean+spread engineered features) doesn't have a clean fixed
    ensemble/variable factorization the way "raw member x raw variable"
    does.
    """

    def __init__(self, in_channels: int, unit: int = 64):
        super().__init__()
        self.unit = unit

        self.conv1 = nn.Sequential(_cbr3d(in_channels, unit), _cbr3d(unit, unit))
        self.pool1 = nn.AvgPool3d(kernel_size=2, stride=2)

        self.conv2 = nn.Sequential(_cbr3d(unit, unit * 2), _cbr3d(unit * 2, unit * 2))
        self.pool2 = nn.AvgPool3d(kernel_size=2, stride=2)

        self.conv3 = nn.Sequential(_cbr3d(unit * 2, unit * 4), _cbr3d(unit * 4, unit * 4))
        self.pool3 = nn.AvgPool3d(kernel_size=2, stride=2)

        self.bottleneck = nn.Sequential(_cbr3d(unit * 4, unit * 8), _cbr3d(unit * 8, unit * 8))

        self.upconv1 = nn.ConvTranspose3d(unit * 8, unit * 4, kernel_size=2, stride=2)
        self.ex_conv1 = nn.Sequential(_cbr3d(unit * 8, unit * 4), _cbr3d(unit * 4, unit * 4))

        self.upconv2 = nn.ConvTranspose3d(unit * 4, unit * 2, kernel_size=2, stride=2)
        self.ex_conv2 = nn.Sequential(_cbr3d(unit * 4, unit * 2), _cbr3d(unit * 2, unit * 2))

        self.upconv3 = nn.ConvTranspose3d(unit * 2, unit, kernel_size=2, stride=2)
        self.ex_conv3 = nn.Sequential(_cbr3d(unit * 2, unit), _cbr3d(unit, unit))

        self.head = nn.Conv3d(unit, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        layer1 = self.conv1(x)
        out = self.pool1(layer1)

        layer2 = self.conv2(out)
        out = self.pool2(layer2)

        layer3 = self.conv3(out)
        out = self.pool3(layer3)

        out = self.bottleneck(out)

        out = self.upconv1(out)
        out = self.ex_conv1(torch.cat([layer3, out], dim=1))

        out = self.upconv2(out)
        out = self.ex_conv2(torch.cat([layer2, out], dim=1))

        out = self.upconv3(out)
        out = self.ex_conv3(torch.cat([layer1, out], dim=1))

        return self.head(out)


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """`mask` is (lat, lon) or broadcastable to `pred`'s shape, 1 = valid."""
    diff2 = (pred - target) ** 2
    mask = mask.to(pred.dtype)
    denom = mask.expand_as(diff2).sum().clamp_min(1.0)
    return (diff2 * mask).sum() / denom


def masked_pattern_correlation_loss(
    pred: torch.Tensor, target: torch.Tensor, lat: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """1 - latitude-weighted Pearson correlation, averaged over batch and
    lead day, computed only over `mask`-valid grid cells.

    `pred`/`target`: (batch, 1, lead, lat, lon).
    `lat`: (lat,) latitude values in degrees.
    `mask`: (lat, lon), 1 = valid IMD cell.
    """
    pred = pred.squeeze(1)   # (batch, lead, lat, lon)
    target = target.squeeze(1)

    weights = torch.cos(torch.deg2rad(lat)).to(pred.dtype).to(pred.device)
    weights = weights.view(1, 1, -1, 1) * mask.to(pred.dtype).unsqueeze(0).unsqueeze(0)

    w_sum = weights.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)

    def w_mean(x):
        return (x * weights).sum(dim=(-2, -1), keepdim=True) / w_sum

    pred_c = pred - w_mean(pred)
    tgt_c = target - w_mean(target)

    cov = (weights * pred_c * tgt_c).sum(dim=(-2, -1))
    var_pred = (weights * pred_c ** 2).sum(dim=(-2, -1)).clamp_min(1e-12)
    var_tgt = (weights * tgt_c ** 2).sum(dim=(-2, -1)).clamp_min(1e-12)

    corr = cov / torch.sqrt(var_pred * var_tgt)
    return (1 - corr).mean()
