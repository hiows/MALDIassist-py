"""Port of src/spectrum_math.cpp.

Nadaraya-Watson Gaussian kernel regression ("KDE") and its 1st-3rd
derivatives, curvature, grid root finding and extrema classification.

The kernel weight for grid point ``xi`` and observation ``x_obs[j]`` is
``k = exp(-0.5 * (xi - x_obs[j])^2 / bw^2) / (bw * sqrt(2*pi))``. The estimate
is the *ratio* ``sum(k * y) / sum(k)`` (kernel regression), and derivatives are
quotient-rule derivatives of that ratio. Observations further than
``cutoff * bw`` (default 5*bw) from ``xi`` are dropped when ``x_obs`` is sorted.
"""

from __future__ import annotations

import math

import numpy as np

SQRT_2PI = 2.50662827463100050242
DEFAULT_KDE_CUTOFF = 5.0
_DENOM_FLOOR = np.finfo(float).tiny


def _is_strictly_increasing(v: np.ndarray) -> bool:
    if v.size < 2:
        return True
    return bool(np.all(np.diff(v) > 0))


def _support_bounds(xi: float, x_obs: np.ndarray, radius: float):
    """Return (j_lo, j_hi) inclusive indices of obs within [xi-radius, xi+radius].

    Mirrors ``kde_support_bounds``: returns an empty range (j_lo > j_hi) when
    no observation falls inside.
    """
    n_obs = x_obs.size
    lo = xi - radius
    hi = xi + radius
    j_lo = int(np.searchsorted(x_obs, lo, side="left"))
    j_hi = int(np.searchsorted(x_obs, hi, side="right")) - 1
    if j_lo > j_hi:
        return n_obs, n_obs - 1
    return j_lo, j_hi


def _seqsum(a):
    """Strict left-to-right accumulation, matching the C++ ``sum += a[j]`` loop.

    ``np.sum``/``np.add.reduce`` use pairwise (blocked) summation which differs
    from R's sequential Rcpp loop at the ULP level. That difference is enough to
    flip knife-edge ties (e.g. symmetric windows in ``find_frequent_mz`` where
    two mirror-image local maxima have equal density under pairwise summation but
    a tiny asymmetry under sequential summation). ``np.add.accumulate`` is a true
    prefix sum, so its last element equals the sequential sum bit-for-bit.
    """
    if a.size == 0:
        return 0.0
    if a.size == 1:
        return float(a[0])
    return float(np.add.accumulate(a)[-1])


def _accumulate(xi, x_obs, y_obs, bw, j_lo, j_hi):
    """Compute (n0,d0,n1,d1,n2,d2,n3,d3) over obs[j_lo:j_hi+1]."""
    bw2 = bw * bw
    bw4 = bw2 * bw2
    bw6 = bw4 * bw2
    inv_bw2 = 1.0 / bw2
    inv_bw4 = 1.0 / bw4
    inv_bw6 = 1.0 / bw6
    kernel_const = 1.0 / (bw * SQRT_2PI)

    xs = x_obs[j_lo:j_hi + 1]
    ys = y_obs[j_lo:j_hi + 1]
    diff = xi - xs
    diff2 = diff * diff
    k = np.exp(-0.5 * diff2 * inv_bw2) * kernel_const
    k1 = -diff * inv_bw2 * k
    k2 = (diff2 * inv_bw4 - inv_bw2) * k
    k3 = diff * (3.0 * bw2 - diff2) * inv_bw6 * k

    n0 = _seqsum(k * ys); d0 = _seqsum(k)
    n1 = _seqsum(k1 * ys); d1 = _seqsum(k1)
    n2 = _seqsum(k2 * ys); d2 = _seqsum(k2)
    n3 = _seqsum(k3 * ys); d3 = _seqsum(k3)
    return n0, d0, n1, d1, n2, d2, n3, d3


def _combine(deriv_order, n0, d0, n1, d1, n2, d2, n3, d3):
    if deriv_order == 0:
        return n0 / d0
    if deriv_order == 1:
        return (n1 * d0 - n0 * d1) / (d0 * d0)
    if deriv_order == 2:
        d0_2 = d0 * d0
        d0_3 = d0_2 * d0
        return n2 / d0 - n0 * d2 / d0_2 - 2.0 * n1 * d1 / d0_2 + 2.0 * n0 * d1 * d1 / d0_3
    d0_2 = d0 * d0
    d0_3 = d0_2 * d0
    d0_4 = d0_3 * d0
    return (n3 / d0 - 3.0 * n2 * d1 / d0_2
            + 3.0 * n1 * (2.0 * d1 * d1 / d0_3 - d2 / d0_2)
            + n0 * (-d3 / d0_2 + 6.0 * d1 * d2 / d0_3 - 6.0 * d1 * d1 * d1 / d0_4))


def kde_deriv_scalar(xi, x_obs, y_obs, bw, deriv_order, use_truncation, cutoff):
    j_lo = 0
    j_hi = x_obs.size - 1
    if use_truncation:
        j_lo, j_hi = _support_bounds(xi, x_obs, cutoff * bw)
        if j_lo > j_hi:
            return np.nan
    n0, d0, n1, d1, n2, d2, n3, d3 = _accumulate(xi, x_obs, y_obs, bw, j_lo, j_hi)
    if d0 <= _DENOM_FLOOR:
        return np.nan
    return _combine(deriv_order, n0, d0, n1, d1, n2, d2, n3, d3)


def gauss_kde_eval(x, x_obs, y_obs, bw, deriv_order,
                   use_truncation=True, cutoff=DEFAULT_KDE_CUTOFF):
    x = np.asarray(x, dtype=float)
    x_obs = np.asarray(x_obs, dtype=float)
    y_obs = np.asarray(y_obs, dtype=float)
    if bw <= 0 or not np.isfinite(bw):
        raise ValueError("bw must be a finite positive number.")
    if deriv_order < 0 or deriv_order > 3:
        raise ValueError("deriv_order must be 0, 1, 2, or 3.")
    sorted_obs = _is_strictly_increasing(x_obs)
    truncate = use_truncation and sorted_obs
    n = x.size
    out = np.empty(n)
    for i in range(n):
        xi = x[i]
        if truncate:
            j_lo, j_hi = _support_bounds(xi, x_obs, cutoff * bw)
            if j_lo > j_hi:
                out[i] = np.nan
                continue
        else:
            j_lo, j_hi = 0, x_obs.size - 1
        n0, d0, n1, d1, n2, d2, n3, d3 = _accumulate(xi, x_obs, y_obs, bw, j_lo, j_hi)
        if d0 <= _DENOM_FLOOR:
            out[i] = np.nan
            continue
        out[i] = _combine(deriv_order, n0, d0, n1, d1, n2, d2, n3, d3)
    return out


def gauss_kde_all_eval(x, x_obs, y_obs, bw,
                       use_truncation=True, cutoff=DEFAULT_KDE_CUTOFF):
    """Return an (n, 4) array with columns kde, d1, d2, d3."""
    x = np.asarray(x, dtype=float)
    x_obs = np.asarray(x_obs, dtype=float)
    y_obs = np.asarray(y_obs, dtype=float)
    if bw <= 0 or not np.isfinite(bw):
        raise ValueError("bw must be a finite positive number.")
    sorted_obs = _is_strictly_increasing(x_obs)
    truncate = use_truncation and sorted_obs
    n = x.size
    out = np.empty((n, 4))
    for i in range(n):
        xi = x[i]
        if truncate:
            j_lo, j_hi = _support_bounds(xi, x_obs, cutoff * bw)
            if j_lo > j_hi:
                out[i, :] = np.nan
                continue
        else:
            j_lo, j_hi = 0, x_obs.size - 1
        n0, d0, n1, d1, n2, d2, n3, d3 = _accumulate(xi, x_obs, y_obs, bw, j_lo, j_hi)
        if d0 <= _DENOM_FLOOR:
            out[i, :] = np.nan
            continue
        d0_2 = d0 * d0
        d0_3 = d0_2 * d0
        d0_4 = d0_3 * d0
        out[i, 0] = n0 / d0
        out[i, 1] = (n1 * d0 - n0 * d1) / d0_2
        out[i, 2] = n2 / d0 - n0 * d2 / d0_2 - 2.0 * n1 * d1 / d0_2 + 2.0 * n0 * d1 * d1 / d0_3
        out[i, 3] = (n3 / d0 - 3.0 * n2 * d1 / d0_2
                     + 3.0 * n1 * (2.0 * d1 * d1 / d0_3 - d2 / d0_2)
                     + n0 * (-d3 / d0_2 + 6.0 * d1 * d2 / d0_3 - 6.0 * d1 * d1 * d1 / d0_4))
    return out


def _bisection_root_kde_deriv(x_obs, y_obs, bw, deriv_order, use_truncation,
                              cutoff, left, right, yi, tol, max_iter):
    f_left = kde_deriv_scalar(left, x_obs, y_obs, bw, deriv_order, use_truncation, cutoff) - yi
    f_right = kde_deriv_scalar(right, x_obs, y_obs, bw, deriv_order, use_truncation, cutoff) - yi
    if not (np.isfinite(f_left) and np.isfinite(f_right)):
        raise ValueError("KDE derivative returned non-finite values during bisection.")
    if abs(f_left) <= tol:
        return left
    if abs(f_right) <= tol:
        return right
    if f_left * f_right > 0.0:
        raise ValueError("Bisection requires a sign-changing interval.")
    mid = 0.5 * (left + right)
    for _ in range(max_iter):
        mid = 0.5 * (left + right)
        f_mid = kde_deriv_scalar(mid, x_obs, y_obs, bw, deriv_order, use_truncation, cutoff) - yi
        if not np.isfinite(f_mid):
            raise ValueError("KDE derivative returned non-finite values during bisection.")
        if abs(f_mid) <= tol or abs(right - left) <= tol:
            return mid
        if f_left * f_mid <= 0.0:
            right = mid
            f_right = f_mid
        else:
            left = mid
            f_left = f_mid
    return mid


def _close_to_existing(roots, candidate, tol):
    for r in roots:
        if abs(r - candidate) <= tol:
            return True
    return False


def find_roots_on_grid(x, y_grid, x_obs, y_obs, bw, yi, tol, max_iter,
                       use_truncation, cutoff):
    x = np.asarray(x, dtype=float)
    y_grid = np.asarray(y_grid, dtype=float)
    n = x.size
    roots = []
    for i in range(n):
        fi = y_grid[i] - yi
        if abs(fi) <= tol and not _close_to_existing(roots, x[i], tol):
            roots.append(x[i])
    for i in range(n - 1):
        f_left = y_grid[i] - yi
        f_right = y_grid[i + 1] - yi
        if f_left == 0.0 or f_right == 0.0:
            continue
        if f_left * f_right < 0.0:
            root = _bisection_root_kde_deriv(
                x_obs, y_obs, bw, 1, use_truncation, cutoff,
                x[i], x[i + 1], yi, tol, max_iter)
            if not _close_to_existing(roots, root, tol):
                roots.append(root)
    roots.sort()
    return np.array(roots, dtype=float)


def find_extrema_from_grid(x, d1_grid, x_obs, y_obs, bw, tol, max_iter,
                           use_truncation=None, cutoff=DEFAULT_KDE_CUTOFF):
    x = np.asarray(x, dtype=float)
    d1_grid = np.asarray(d1_grid, dtype=float)
    x_obs = np.asarray(x_obs, dtype=float)
    y_obs = np.asarray(y_obs, dtype=float)
    if use_truncation is None:
        use_truncation = _is_strictly_increasing(x_obs)
    x_roots = find_roots_on_grid(x, d1_grid, x_obs, y_obs, bw, 0.0, tol, max_iter,
                                 use_truncation, cutoff)
    if x_roots.size == 0:
        return {"local_min": np.array([np.nan]),
                "local_max": np.array([np.nan]),
                "plateau": np.array([np.nan])}
    d2_roots = np.array([
        kde_deriv_scalar(xr, x_obs, y_obs, bw, 2, use_truncation, cutoff)
        for xr in x_roots
    ])
    if np.any(~np.isfinite(d2_roots)):
        raise ValueError("Second KDE derivative non-finite at root positions.")
    local_max = x_roots[d2_roots < -tol]
    local_min = x_roots[d2_roots > tol]
    plateau = x_roots[np.abs(d2_roots) <= tol]
    return {
        "local_min": local_min if local_min.size else np.array([np.nan]),
        "local_max": local_max if local_max.size else np.array([np.nan]),
        "plateau": plateau if plateau.size else np.array([np.nan]),
    }


def curvature_from_derivs(d1, d2, absolute=True):
    d1 = np.asarray(d1, dtype=float)
    d2 = np.asarray(d2, dtype=float)
    if d1.size != d2.size:
        raise ValueError("d1 and d2 must have the same length.")
    denom = np.power(1.0 + d1 * d1, 1.5)
    numer = np.abs(d2) if absolute else d2
    return numer / denom
