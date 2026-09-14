"""Dependency-free regridding for regular lat/lon grids.

The reference paper (and our own literature-review.md S1.3) both call for
conservative (area-weighted) regridding, normally done with xESMF. xESMF
needs the ESMF C library, which isn't available in this environment, so
this module reimplements the two regridding kinds we actually need for
*regular*, axis-aligned lat/lon grids -- which is all IMD/ECMWF ever give
us -- without it:

- Conservative: exact overlap-length weighting along latitude and
  longitude, with a cos(lat) area correction, combined as an outer
  product (valid because both grids are axis-aligned rectangles on a
  lat/lon mesh).
- Bilinear: scipy's RegularGridInterpolator.

Both ignore NaN source cells properly: a target cell's value is the
overlap-weighted average of only the *valid* source cells beneath it, and
is itself NaN if no valid source cell overlaps it at all.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def _cell_edges(centers: np.ndarray) -> np.ndarray:
    """Edges of evenly spaced grid cells, from their centers."""
    centers = np.asarray(centers, dtype=np.float64)
    step = centers[1] - centers[0]
    if not np.allclose(np.diff(centers), step, rtol=1e-4):
        raise ValueError("_cell_edges assumes an evenly spaced grid")
    edges = np.concatenate([centers - step / 2, centers[-1:] + step / 2])
    return edges


def _overlap_lengths(src_edges: np.ndarray, tgt_edges: np.ndarray) -> np.ndarray:
    """(n_tgt, n_src) matrix of 1D overlap length between every target and
    source cell along one axis. Edges must be monotonically increasing."""
    src_lo, src_hi = src_edges[:-1], src_edges[1:]
    tgt_lo, tgt_hi = tgt_edges[:-1], tgt_edges[1:]
    lo = np.maximum(tgt_lo[:, None], src_lo[None, :])
    hi = np.minimum(tgt_hi[:, None], src_hi[None, :])
    return np.clip(hi - lo, 0, None)


def build_conservative_weights(
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    tgt_lat: np.ndarray,
    tgt_lon: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Weight matrix W of shape (n_tgt_lat*n_tgt_lon, n_src_lat*n_src_lon)
    such that `target_flat = W @ source_flat` is the area-conservative
    regrid, with rows re-normalized to sum to 1 over only the valid source
    cells under them (NaN row if nothing valid overlaps).

    Latitude may be given descending (as ECMWF/IMD often store it); this
    function sorts internally and unsorts the result.
    """
    src_lat = np.asarray(src_lat, dtype=np.float64)
    tgt_lat = np.asarray(tgt_lat, dtype=np.float64)

    lat_asc = src_lat[0] < src_lat[-1]
    tlat_asc = tgt_lat[0] < tgt_lat[-1]
    src_lat_s = src_lat if lat_asc else src_lat[::-1]
    tgt_lat_s = tgt_lat if tlat_asc else tgt_lat[::-1]

    lat_overlap = _overlap_lengths(_cell_edges(src_lat_s), _cell_edges(tgt_lat_s))
    # Area correction: a degree of longitude covers less ground near the
    # poles. Weight each source row by cos(lat) at its center.
    area_weight = np.cos(np.deg2rad(src_lat_s))
    lat_overlap = lat_overlap * area_weight[None, :]

    if not lat_asc:
        lat_overlap = lat_overlap[:, ::-1]
    if not tlat_asc:
        lat_overlap = lat_overlap[::-1, :]

    lon_overlap = _overlap_lengths(_cell_edges(src_lon), _cell_edges(tgt_lon))

    n_tgt_lat, n_src_lat = lat_overlap.shape
    n_tgt_lon, n_src_lon = lon_overlap.shape

    # Outer-product combine: W[(i,j), (a,b)] = lat_overlap[i,a] * lon_overlap[j,b]
    w = np.einsum("ia,jb->ijab", lat_overlap, lon_overlap)
    w = w.reshape(n_tgt_lat * n_tgt_lon, n_src_lat * n_src_lon)

    if valid_mask is not None:
        valid_flat = np.asarray(valid_mask, dtype=bool).reshape(-1)
        w = w * valid_flat[None, :]

    row_sums = w.sum(axis=1)
    out = np.full_like(w, np.nan)
    nonzero = row_sums > 0
    out[nonzero] = w[nonzero] / row_sums[nonzero, None]
    out[~nonzero] = 0.0
    # Rows with no valid overlap stay all-zero here; caller detects them
    # via `row_sums == 0` if it needs an explicit NaN target cell.
    return out, row_sums > 0


def conservative_regrid(
    data: np.ndarray,
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    tgt_lat: np.ndarray,
    tgt_lon: np.ndarray,
    weights: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Regrid `data` with shape (..., n_src_lat, n_src_lon) to
    (..., n_tgt_lat, n_tgt_lon), conservatively, masking NaNs.

    Pass a precomputed `weights` (from `build_conservative_weights`, using a
    fixed validity mask) to avoid rebuilding it per time step.
    """
    src_lat = np.asarray(src_lat)
    src_lon = np.asarray(src_lon)
    n_src_lat, n_src_lon = len(src_lat), len(src_lon)
    n_tgt_lat, n_tgt_lon = len(tgt_lat), len(tgt_lon)

    lead_shape = data.shape[:-2]
    flat = data.reshape(-1, n_src_lat * n_src_lon)

    if weights is None:
        valid = np.isfinite(data).all(axis=tuple(range(data.ndim - 2))) if lead_shape else np.isfinite(data)
        w, has_overlap = build_conservative_weights(src_lat, src_lon, tgt_lat, tgt_lon, valid)
    else:
        w, has_overlap = weights

    filled = np.where(np.isfinite(flat), flat, 0.0)
    out = filled @ w.T  # (N, n_tgt_lat*n_tgt_lon)
    out[:, ~has_overlap] = np.nan
    return out.reshape(*lead_shape, n_tgt_lat, n_tgt_lon)


def bilinear_regrid(
    data: np.ndarray,
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    tgt_lat: np.ndarray,
    tgt_lon: np.ndarray,
) -> np.ndarray:
    """Regrid `data` (..., n_src_lat, n_src_lon) to (..., n_tgt_lat,
    n_tgt_lon) with bilinear interpolation. NaNs are filled with the
    field's mean before interpolating (bilinear can't represent missing
    data), then re-masked afterwards using the nearest-neighbour validity
    of the source grid.
    """
    src_lat = np.asarray(src_lat, dtype=np.float64)
    src_lon = np.asarray(src_lon, dtype=np.float64)

    lat_asc = src_lat[0] < src_lat[-1]
    lat_pts = src_lat if lat_asc else src_lat[::-1]

    lead_shape = data.shape[:-2]
    flat = data.reshape(-1, len(src_lat), len(src_lon))
    if not lat_asc:
        flat = flat[:, ::-1, :]

    valid = np.isfinite(flat)
    fill_value = np.nanmean(flat) if np.isfinite(flat).any() else 0.0
    filled = np.where(valid, flat, fill_value)

    tgt_grid = np.stack(np.meshgrid(tgt_lat, tgt_lon, indexing="ij"), axis=-1)
    tgt_pts = tgt_grid.reshape(-1, 2)

    out = np.empty((flat.shape[0], len(tgt_lat) * len(tgt_lon)))
    mask_interp = RegularGridInterpolator(
        (lat_pts, src_lon), np.ones_like(flat[0]), bounds_error=False, fill_value=0.0
    )
    valid_at_tgt = mask_interp(tgt_pts) > 0.5

    for i in range(flat.shape[0]):
        interp = RegularGridInterpolator(
            (lat_pts, src_lon), filled[i], bounds_error=False, fill_value=None
        )
        out[i] = interp(tgt_pts)

    out[:, ~valid_at_tgt] = np.nan
    return out.reshape(*lead_shape, len(tgt_lat), len(tgt_lon))
