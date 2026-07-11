"""Port of the peak-detection layer.

R sources: find_peaks_fast.R (+ cpp_find_peaks_fast.cpp), find_peaks.R,
find_extrema.R, find_extrema_fast.R, get_gauss_kde.R, build_kde_spectrum.R,
get_curvature_fun.R, estimate_peak_boundaries.R.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd

from . import spectrum_math as _sm
from ._util import as_xy, is_single_spectrum, output_colnames
from .metrics import estimate_peak_strength
from .savgol import cpp_savitzky_golay_filter
from .rcompat import quantile_type7


# ---------------------------------------------------------------------------
# get_gauss_kde / build_kde_spectrum / get_curvature_fun
# ---------------------------------------------------------------------------
def get_gauss_kde(data, bw: float = 1.0, d: int = 0):
    x_obs, y_obs, _ = as_xy(data)
    x_obs = np.ascontiguousarray(x_obs, dtype=float)
    y_obs = np.ascontiguousarray(y_obs, dtype=float)
    if bw <= 0:
        raise ValueError("'bw' must be greater than 0.")
    d = int(d)
    if d not in (0, 1, 2, 3):
        raise ValueError("'d' must be one of 0, 1, 2, or 3.")

    def fun(x):
        x = np.asarray(x, dtype=float)
        return _sm.gauss_kde_eval(x, x_obs, y_obs, bw, d)

    return fun


def build_kde_spectrum(spectrum, bw: float | None = None):
    x, y, _ = as_xy(spectrum)
    if bw is None:
        bw = float(np.median(np.diff(x)))
    cols = output_colnames(spectrum)
    f = get_gauss_kde(spectrum, bw=bw, d=0)
    y_kde = f(x)
    kde_spectrum = pd.DataFrame({cols[0]: x, cols[1]: y_kde})
    return {"spectrum": kde_spectrum, "gauss_kde": f, "bw": bw}


def build_kde_spectra(spectra, bw: float | None = None, n_cores: int = 1):
    if is_single_spectrum(spectra):
        return build_kde_spectrum(spectra, bw=bw)
    if isinstance(spectra, dict):
        return OrderedDict((k, build_kde_spectrum(v, bw=bw)) for k, v in spectra.items())
    return [build_kde_spectrum(v, bw=bw) for v in spectra]


def get_curvature_fun(first_deriv, second_deriv, absolute: bool = True):
    def fun(x):
        x = np.asarray(x, dtype=float)
        d1 = np.asarray(first_deriv(x), dtype=float)
        d2 = np.asarray(second_deriv(x), dtype=float)
        return _sm.curvature_from_derivs(d1, d2, absolute=absolute)

    return fun


# ---------------------------------------------------------------------------
# find_extrema (root-based) and find_extrema_fast (sign-change)
# ---------------------------------------------------------------------------
def find_extrema(first_deriv, second_deriv, x, tol: float = 1e-5, max_iter: int = 100):
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        raise ValueError("'x' must contain at least two values.")
    d1_grid = np.asarray(first_deriv(x), dtype=float)
    # generic root finder on the grid using the supplied first-derivative fn
    roots = _find_roots_generic(first_deriv, x, 0.0, tol, int(max_iter))
    if roots.size == 0:
        return {"local_min": np.array([np.nan]),
                "local_max": np.array([np.nan]),
                "plateau": np.array([np.nan])}
    d2 = np.asarray(second_deriv(roots), dtype=float)
    local_max = roots[d2 < -tol]
    local_min = roots[d2 > tol]
    plateau = roots[np.abs(d2) <= tol]
    return {
        "local_min": local_min if local_min.size else np.array([np.nan]),
        "local_max": local_max if local_max.size else np.array([np.nan]),
        "plateau": plateau if plateau.size else np.array([np.nan]),
    }


def _bisection_root_cached(fun, left, right, yi, tol, max_iter):
    def fx(v):
        return float(fun(np.array([v]))[0])

    f_left = fx(left) - yi
    f_right = fx(right) - yi
    if abs(f_left) <= tol:
        return left
    if abs(f_right) <= tol:
        return right
    if f_left * f_right > 0.0:
        raise ValueError("Bisection requires a sign-changing interval.")
    mid = 0.5 * (left + right)
    for _ in range(max_iter):
        mid = 0.5 * (left + right)
        f_mid = fx(mid) - yi
        if abs(f_mid) <= tol or abs(right - left) <= tol:
            return mid
        if f_left * f_mid <= 0.0:
            right = mid
            f_right = f_mid
        else:
            left = mid
            f_left = f_mid
    return mid


def _find_roots_generic(fun, x, yi, tol, max_iter):
    x = np.asarray(x, dtype=float)
    y_grid = np.asarray(fun(x), dtype=float)
    roots = []

    def close(cand):
        for r in roots:
            if abs(r - cand) <= tol:
                return True
        return False

    for i in range(x.size):
        if abs(y_grid[i] - yi) <= tol and not close(x[i]):
            roots.append(x[i])
    for i in range(x.size - 1):
        fl = y_grid[i] - yi
        fr = y_grid[i + 1] - yi
        if fl == 0.0 or fr == 0.0:
            continue
        if fl * fr < 0.0:
            root = _bisection_root_cached(fun, x[i], x[i + 1], yi, tol, max_iter)
            if not close(root):
                roots.append(root)
    roots.sort()
    return np.array(roots, dtype=float)


def find_extrema_fast(x, y, plateau: str = "middle", na_rm: bool = True):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size != y.size:
        raise ValueError("'x' and 'y' must have the same length.")
    if na_rm:
        keep = np.isfinite(x) & np.isfinite(y)
        x = x[keep]
        y = y[keep]
    if y.size < 3:
        raise ValueError("At least 3 points are required.")

    dy = np.diff(y)
    s = np.sign(dy)
    ls = s.size

    sharp_max = np.where((s[:-1] > 0) & (s[1:] < 0))[0] + 1
    sharp_min = np.where((s[:-1] < 0) & (s[1:] > 0))[0] + 1
    local_max_idx = list(sharp_max)
    local_min_idx = list(sharp_min)

    zero = (s == 0)
    if np.any(zero):
        # run-length encode boolean `zero`
        idx = 0
        while idx < ls:
            if zero[idx]:
                start_s = idx
                end_s = idx
                while end_s + 1 < ls and zero[end_s + 1]:
                    end_s += 1
                left_s_idx = start_s - 1
                right_s_idx = end_s + 1
                if left_s_idx >= 0 and right_s_idx < ls:
                    left_sign = s[left_s_idx]
                    right_sign = s[right_s_idx]
                    start_y = start_s
                    end_y = end_s + 1
                    if plateau == "first":
                        idx_one = start_y
                    elif plateau == "last":
                        idx_one = end_y
                    else:
                        idx_one = int(round((start_y + end_y) / 2.0))
                    if left_sign > 0 and right_sign < 0:
                        local_max_idx.append(idx_one)
                    if left_sign < 0 and right_sign > 0:
                        local_min_idx.append(idx_one)
                idx = end_s + 1
            else:
                idx += 1

    local_max_idx = sorted(set(local_max_idx))
    local_min_idx = sorted(set(local_min_idx))
    local_max = x[local_max_idx] if local_max_idx else np.array([np.nan])
    local_min = x[local_min_idx] if local_min_idx else np.array([np.nan])
    return {"local_min": local_min, "local_max": local_max}


# ---------------------------------------------------------------------------
# find_peaks_fast
# ---------------------------------------------------------------------------
_PLATEAU_ID = {"middle": 0, "first": 1, "last": 2}


def _find_peaks_fast_core(x, y, hws_peaks, plateau_id):
    n = x.size
    localmax_idx = []
    i = 1
    while i < n - 1:
        if y[i] > y[i - 1] and y[i] > y[i + 1]:
            localmax_idx.append(i)
            i += 1
            continue
        if y[i] > y[i - 1] and y[i] == y[i + 1]:
            start = i
            end = i + 1
            while end + 1 < n and y[end] == y[end + 1]:
                end += 1
            if end < n - 1 and y[end] > y[end + 1]:
                if plateau_id == 1:
                    idx_selected = start
                elif plateau_id == 2:
                    idx_selected = end
                else:
                    idx_selected = start + (end - start) // 2
                localmax_idx.append(idx_selected)
            i = end + 1
            continue
        i += 1

    if not localmax_idx:
        return np.zeros(0), np.zeros(0)

    lm_x = x[localmax_idx]
    lm_y = y[localmax_idx]
    m = lm_x.size

    keep = []
    for a in range(m):
        left = lm_x[a] - hws_peaks
        right = lm_x[a] + hws_peaks
        window = lm_y[(lm_x >= left) & (lm_x <= right)]
        max_y = np.max(window)
        if lm_y[a] == max_y:
            keep.append(a)
    keep = np.array(keep, dtype=int)
    return lm_x[keep], lm_y[keep]


def find_peaks_fast(data, hws_peaks: float = 10.0, plateau: str = "middle",
                    na_rm: bool = True) -> pd.DataFrame:
    if plateau not in _PLATEAU_ID:
        raise ValueError("'plateau' must be 'middle', 'first', or 'last'.")
    x, y, _ = as_xy(data)
    cols = _peak_colnames(data)
    if na_rm:
        keep = np.isfinite(x) & np.isfinite(y)
        x = x[keep]
        y = y[keep]
    if x.size < 3:
        raise ValueError("At least 3 points are required.")
    px, py = _find_peaks_fast_core(x, y, float(hws_peaks), _PLATEAU_ID[plateau])
    df = pd.DataFrame({cols[0]: px, cols[1]: py})
    df["type"] = "peak"
    return df


def _peak_colnames(data):
    if isinstance(data, pd.DataFrame) and len(data.columns) >= 2:
        cols = list(data.columns[:2])
        if all(isinstance(c, str) and c != "" for c in cols):
            return cols
    return ["x", "y"]


# ---------------------------------------------------------------------------
# find_peaks (ordinary + shoulder)
# ---------------------------------------------------------------------------
def find_peaks(data, bw: float | None = None, hws_peaks: float = 10.0,
               merge_tol: float | None = None, tol: float = 1e-5,
               max_iter: int = 100, weight_type: str = "raw",
               hws_grid_kappa_smooth=range(3, 21),
               cutoff_kappa_peak_strength: float = 0.5,
               peak_retention_fraction: float = 0.25) -> pd.DataFrame:
    x, y, _ = as_xy(data)
    if x.size < 3:
        raise ValueError("'data' must contain at least three rows.")
    if bw is None:
        bw = float(np.median(np.diff(x)))
    if bw <= 0:
        raise ValueError("'bw' must be a positive finite numeric scalar.")
    if hws_peaks <= 0:
        raise ValueError("'hws_peaks' must be positive.")
    if merge_tol is None:
        merge_tol = hws_peaks
    if weight_type not in ("raw", "sqrt", "log10", "none"):
        raise ValueError("invalid 'weight_type'.")

    hws_grid = sorted(set(int(h) for h in hws_grid_kappa_smooth))
    if any(h < 2 for h in hws_grid):
        raise ValueError("'hws_grid_kappa_smooth' must contain integers >= 2.")
    if any((2 * h + 1) > x.size for h in hws_grid):
        raise ValueError("SG window sizes must be <= nrow(data).")

    cols = output_colnames(data)
    x_obs = np.ascontiguousarray(x, dtype=float)
    y_obs = np.ascontiguousarray(y, dtype=float)

    # 4. KDE + derivatives on the grid
    kde_all = _sm.gauss_kde_all_eval(x_obs, x_obs, y_obs, bw)
    y_est = kde_all[:, 0]
    d1_grid = kde_all[:, 1]
    d2_grid = kde_all[:, 2]

    # 5. weighted reverse-signed curvature
    if weight_type == "none":
        weights = np.ones_like(y_est)
    elif weight_type == "raw":
        weights = np.maximum(y_est, 0.0)
    elif weight_type == "sqrt":
        weights = np.sqrt(np.maximum(y_est, 0.0))
    else:
        weights = np.log10(np.maximum(y_est, 0.0) + 1.0)

    curvature = _sm.curvature_from_derivs(d1_grid, d2_grid, absolute=False)
    weighted_kappa = np.abs(np.minimum(0.0, curvature)) * weights

    # 6. smooth + average
    stack = [weighted_kappa]
    for hws_i in hws_grid:
        sm = cpp_savitzky_golay_filter(weighted_kappa, hws_i, 3)
        stack.append(np.maximum(0.0, sm))
    avg_kappa = np.mean(np.vstack(stack), axis=0)

    data_kappa = pd.DataFrame({"x": x, "kappa": avg_kappa})

    # 7. shoulder candidates
    peaks_kappa = find_peaks_fast(data_kappa, hws_peaks=hws_peaks,
                                  plateau="middle", na_rm=True)
    if peaks_kappa.shape[0] == 0:
        x_shoulder_candidates = np.zeros(0)
    else:
        strength = estimate_peak_strength(data_kappa, peaks_kappa, k=1,
                                          normalization_type="raw")
        valid = np.isfinite(strength)
        if not np.any(valid):
            x_shoulder_candidates = np.zeros(0)
        else:
            pk_x = peaks_kappa.iloc[:, 0].to_numpy()[valid]
            strength = strength[valid]
            cutoff = min(cutoff_kappa_peak_strength,
                         quantile_type7(strength, 1.0 - peak_retention_fraction))
            x_shoulder_candidates = pk_x[strength > cutoff]

    # 8. ordinary KDE peaks
    extrema = _sm.find_extrema_from_grid(x_obs, d1_grid, x_obs, y_obs, bw,
                                         tol, int(max_iter))
    x_localmax = extrema["local_max"]
    if x_localmax.size == 1 and np.isnan(x_localmax[0]):
        x_localmax = np.zeros(0)

    # 9. remove shoulder candidates near ordinary peaks
    if x_shoulder_candidates.size and x_localmax.size:
        overlaps = np.array([
            np.any(np.abs(xi - x_localmax) <= merge_tol)
            for xi in x_shoulder_candidates
        ])
        x_shoulder = x_shoulder_candidates[~overlaps]
    else:
        x_shoulder = x_shoulder_candidates

    # 10. combine
    def kde_at(xp):
        if xp.size == 0:
            return np.zeros(0)
        return _sm.gauss_kde_eval(xp, x_obs, y_obs, bw, 0)

    peak_x = np.concatenate([x_localmax, x_shoulder])
    peak_y = np.concatenate([kde_at(x_localmax), kde_at(x_shoulder)])
    peak_type = (["peak"] * x_localmax.size) + (["shoulder"] * x_shoulder.size)

    if peak_x.size == 0:
        return _empty_peaks(cols)

    order = np.argsort(peak_x, kind="stable")
    peak_x = peak_x[order]
    peak_y = peak_y[order]
    peak_type = [peak_type[i] for i in order]

    # 11. remove weaker nearby candidates
    n_peaks = peak_x.size
    type_priority = np.array([1 if t == "peak" else 2 for t in peak_type])
    keep = np.zeros(n_peaks, dtype=bool)
    for i in range(n_peaks):
        nearby = np.where((peak_x >= peak_x[i] - merge_tol) &
                          (peak_x <= peak_x[i] + merge_tol))[0]
        nearby_y = peak_y[nearby]
        max_y = np.max(nearby_y)
        strongest = nearby[nearby_y == max_y]
        if strongest.size == 1:
            keep[i] = (i == strongest[0])
        else:
            pref_priority = np.min(type_priority[strongest])
            preferred = strongest[type_priority[strongest] == pref_priority]
            selected = preferred[int(np.argmin(peak_x[preferred]))]
            keep[i] = (i == selected)

    peak_x = peak_x[keep]
    peak_y = peak_y[keep]
    peak_type = [t for t, k in zip(peak_type, keep) if k]

    order2 = np.argsort(peak_x, kind="stable")
    peak_x = peak_x[order2]
    peak_y = peak_y[order2]
    peak_type = [peak_type[i] for i in order2]

    df = pd.DataFrame({cols[0]: peak_x, cols[1]: peak_y, "type": peak_type})
    return df


def _empty_peaks(cols):
    df = pd.DataFrame({cols[0]: np.zeros(0), cols[1]: np.zeros(0),
                       "type": pd.Series([], dtype=object)})
    return df


# ---------------------------------------------------------------------------
# spectra-level wrappers
# ---------------------------------------------------------------------------
def find_peaks_spectra(spectra, bw=None, hws_peaks=10.0, merge_tol=None,
                       tol=1e-5, max_iter=100, weight_type="raw",
                       hws_grid_kappa_smooth=range(3, 21),
                       cutoff_kappa_peak_strength=0.5,
                       peak_retention_fraction=0.25, n_cores=1):
    kwargs = dict(hws_peaks=hws_peaks, merge_tol=merge_tol, tol=tol,
                  max_iter=max_iter, weight_type=weight_type,
                  hws_grid_kappa_smooth=hws_grid_kappa_smooth,
                  cutoff_kappa_peak_strength=cutoff_kappa_peak_strength,
                  peak_retention_fraction=peak_retention_fraction)
    if bw is not None:
        kwargs["bw"] = bw
    if is_single_spectrum(spectra):
        return find_peaks(spectra, **kwargs)
    if isinstance(spectra, dict):
        return OrderedDict((k, find_peaks(v, **kwargs)) for k, v in spectra.items())
    return [find_peaks(v, **kwargs) for v in spectra]


def find_peaks_spectra_fast(spectra, hws_peaks=10.0, plateau="middle",
                            na_rm=True, n_cores=1):
    if is_single_spectrum(spectra):
        return find_peaks_fast(spectra, hws_peaks=hws_peaks, plateau=plateau, na_rm=na_rm)
    if isinstance(spectra, dict):
        return OrderedDict((k, find_peaks_fast(v, hws_peaks=hws_peaks, plateau=plateau,
                                               na_rm=na_rm)) for k, v in spectra.items())
    return [find_peaks_fast(v, hws_peaks=hws_peaks, plateau=plateau, na_rm=na_rm)
            for v in spectra]


# ---------------------------------------------------------------------------
# estimate_peak_boundaries
# ---------------------------------------------------------------------------
def estimate_peak_boundaries(fun, x, peaks) -> pd.DataFrame:
    x = np.asarray(x, dtype=float)
    px, py, _ = as_xy(peaks)
    cols = _peak_colnames(peaks)
    if px.size > 0:
        order = np.argsort(px, kind="mergesort")
        px = px[order]
        py = py[order]
    if px.size < 2:
        return pd.DataFrame({cols[0]: np.zeros(0), cols[1]: np.zeros(0)})

    y_fun = np.asarray(fun(x), dtype=float)
    n_peaks = px.size
    x_left = px[:-1]
    y_left = py[:-1]
    x_right = px[1:]
    y_right = py[1:]

    def weighted_boundary(xl, xr, wl, wr):
        denom = wl + wr
        if not np.isfinite(denom) or denom <= 0:
            return (xl + xr) / 2.0
        return xl + (xr - xl) * wl / denom

    x_bound_weighted = np.empty(n_peaks - 1)
    for i in range(n_peaks - 1):
        yl = max(0.0, y_left[i])
        yr = max(0.0, y_right[i])
        b_raw = weighted_boundary(x_left[i], x_right[i], yl, yr)
        b_sqrt = weighted_boundary(x_left[i], x_right[i], np.sqrt(yl), np.sqrt(yr))
        b_log = weighted_boundary(x_left[i], x_right[i],
                                  np.log10(1 + yl), np.log10(1 + yr))
        x_bound_weighted[i] = np.mean([b_raw, b_sqrt, b_log])

    y_bound_weighted = np.asarray(fun(x_bound_weighted), dtype=float)

    x_bound = np.empty(n_peaks - 1)
    for i in range(n_peaks - 1):
        keep = (x >= x_left[i]) & (x <= x_right[i])
        if not np.any(keep):
            x_bound[i] = x_bound_weighted[i]
            continue
        x_range = x[keep]
        y_range = y_fun[keep]
        vidx = int(np.argmin(y_range))
        x_valley = x_range[vidx]
        y_valley = y_range[vidx]
        if y_valley <= y_bound_weighted[i] * 0.5:
            x_bound[i] = x_valley
        else:
            x_bound[i] = x_bound_weighted[i]

    y_bound = np.asarray(fun(x_bound), dtype=float) / 2.0
    return pd.DataFrame({cols[0]: x_bound, cols[1]: y_bound})
