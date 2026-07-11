"""Port of R/calculate_peak_prominence.R + src/cpp_calculate_peak_prominence.cpp
and R/estimate_peak_strength.R + src/cpp_estimate_peak_strength.cpp.
"""

from __future__ import annotations

import math

import numpy as np

from ._util import as_xy
from .rcompat import quantile_type7

_EPS = np.finfo(float).eps


def _median(vals: np.ndarray) -> float:
    if vals.size == 0:
        return float("nan")
    return float(np.median(vals))


def _mad(vals: np.ndarray, constant: float = 1.4826) -> float:
    if vals.size == 0:
        return float("nan")
    med = _median(vals)
    if not math.isfinite(med):
        return float("nan")
    mad_raw = _median(np.abs(vals - med))
    if not math.isfinite(mad_raw):
        return float("nan")
    return constant * mad_raw


def _positive_values(vals: np.ndarray) -> np.ndarray:
    return vals[np.isfinite(vals) & (vals > 0.0)]


def calculate_peak_prominence(data, peaks, valley_type: str = "higher",
                              zero_tol: float | None = None) -> np.ndarray:
    if zero_tol is None:
        zero_tol = math.sqrt(_EPS)
    if valley_type not in ("higher", "lower"):
        raise ValueError("'valley_type' must be either 'higher' or 'lower'.")

    x, y, _ = as_xy(data)
    order = np.argsort(x, kind="mergesort")
    x = x[order]
    y = y[order]

    px, py, _ = as_xy(peaks)
    if px.size == 0:
        return np.zeros(0)
    porder = np.argsort(px, kind="mergesort")
    px = px[porder]
    py = py[porder]

    n_peaks = px.size
    x_min = x[0]
    x_max = x[-1]
    prominence = np.empty(n_peaks)

    for i in range(n_peaks):
        xi_peak = px[i]
        yi_peak = py[i]

        # nearest higher peak on the left
        x_left_bound = x_min
        mask_left = (py > yi_peak) & (px < xi_peak)
        if np.any(mask_left):
            x_left_bound = float(np.max(px[mask_left]))

        # nearest higher peak on the right
        x_right_bound = x_max
        mask_right = (py > yi_peak) & (px > xi_peak)
        if np.any(mask_right):
            x_right_bound = float(np.min(px[mask_right]))

        left_region = y[(x >= x_left_bound) & (x <= xi_peak)]
        y_left_valley = float(np.min(left_region)) if left_region.size else yi_peak

        right_region = y[(x >= xi_peak) & (x <= x_right_bound)]
        y_right_valley = float(np.min(right_region)) if right_region.size else yi_peak

        if valley_type == "higher":
            reference_valley = max(y_left_valley, y_right_valley)
        else:
            reference_valley = min(y_left_valley, y_right_valley)

        prom = yi_peak - reference_valley
        if abs(prom) <= zero_tol:
            prom = 0.0
        if prom < 0.0:
            prom = 0.0
        prominence[i] = prom

    # restore original peak order
    out = np.empty(n_peaks)
    out[porder] = prominence
    return out


def _transform_intensity(y: np.ndarray, normalization: str) -> np.ndarray:
    if normalization == "raw":
        return y
    if normalization == "sqrt":
        return np.sqrt(y)
    if normalization == "log10":
        return np.log10(1.0 + y)
    raise ValueError("'normalization' must be one of 'raw', 'sqrt', or 'log10'.")


def estimate_peak_strength(data, peaks, k: float = 1.0,
                           normalization_type: str = "raw") -> np.ndarray:
    if normalization_type not in ("raw", "sqrt", "log10"):
        raise ValueError("'normalization_type' must be 'raw', 'sqrt', or 'log10'.")
    if k < 0:
        raise ValueError("'k' must be >= 0.")

    x_raw, y_raw, _ = as_xy(data)
    n = x_raw.size
    if n < 3:
        raise ValueError("'data' must contain at least 3 rows.")

    x = x_raw
    y = _transform_intensity(y_raw, normalization_type)

    yy_global = _positive_values(y)
    if yy_global.size == 0:
        return np.zeros(0)

    global_strength = _median(yy_global) + _mad(yy_global) * k
    if not math.isfinite(global_strength) or global_strength <= _EPS:
        global_strength = quantile_type7(yy_global, 0.95)
    if not math.isfinite(global_strength) or global_strength <= _EPS:
        global_strength = _EPS

    px, py, _ = as_xy(peaks)
    if px.size == 0:
        return np.zeros(0)
    porder = np.argsort(px, kind="mergesort")
    px_s = px[porder]

    boundary_x = np.concatenate(([x[0]], px_s, [x[-1]]))
    n_with_boundary = boundary_x.size
    peak_strength = np.full(px_s.size, np.nan)

    for i in range(1, n_with_boundary - 1):
        xi = boundary_x[i]

        left_mask = (x > boundary_x[i - 1]) & (x < xi)
        if not np.any(left_mask):
            peak_strength[i - 1] = np.nan
            continue
        left_x = x[left_mask]
        left_y = y[left_mask]
        x_left_valley = left_x[int(np.argmin(left_y))]

        right_mask = (x > xi) & (x < boundary_x[i + 1])
        if not np.any(right_mask):
            peak_strength[i - 1] = np.nan
            continue
        right_x = x[right_mask]
        right_y = y[right_mask]
        x_right_valley = right_x[int(np.argmin(right_y))]

        local_mask = (x > x_left_valley) & (x < x_right_valley) & np.isfinite(y)
        yy_local = y[local_mask]
        if yy_local.size == 0:
            peak_strength[i - 1] = np.nan
            continue

        local_strength = float(np.mean(yy_local)) + _mad(yy_local) * k
        if not math.isfinite(local_strength) or local_strength <= _EPS:
            yy_local_pos = _positive_values(yy_local)
            if yy_local_pos.size == 0:
                local_strength = _EPS
            else:
                local_strength = quantile_type7(yy_local_pos, 0.95)
        if not math.isfinite(local_strength) or local_strength <= _EPS:
            local_strength = _EPS

        strength_i = local_strength / (global_strength + local_strength)
        peak_strength[i - 1] = strength_i if math.isfinite(strength_i) else np.nan

    out = np.full(px_s.size, np.nan)
    out[porder] = peak_strength
    return out
