"""Port of src/cpp_savitzkyGolay_filter.cpp.

Savitzky-Golay smoothing with boundary-specific coefficients. Center points use
the symmetric fit; the first/last ``hws`` points use boundary coefficient rows
(the top rows are computed directly, the bottom rows are mirror images).
"""

from __future__ import annotations

import numpy as np

_coef_cache: dict[tuple[int, int], np.ndarray] = {}


def _calculate_savgol_coef_row(window_size: int, pno: int, target_index: int) -> np.ndarray:
    """One row of SG smoothing coefficients estimating the value at ``target_index``.

    Mirrors ``calculate_savgol_coef_row``: build the Vandermonde matrix ``X``
    with ``X[r, c] = offset^c`` (offset = r - target_index), solve
    ``(X^T X) beta = e0`` and return ``coef[r] = sum_c beta[c] * X[r, c]``.
    """
    n_coef = pno + 1
    offsets = np.arange(window_size, dtype=float) - target_index
    X = np.vander(offsets, N=n_coef, increasing=True)  # X[r, c] = offset^c
    XtX = X.T @ X
    e0 = np.zeros(n_coef)
    e0[0] = 1.0
    beta = np.linalg.solve(XtX, e0)
    coef = X @ beta
    return coef


def _build_coef(hws: int, pno: int) -> np.ndarray:
    key = (hws, pno)
    cached = _coef_cache.get(key)
    if cached is not None:
        return cached
    window_size = 2 * hws + 1
    coef = np.zeros((window_size, window_size))
    for i in range(hws + 1):
        coef[i] = _calculate_savgol_coef_row(window_size, pno, i)
    for i in range(hws + 1, window_size):
        mirror_row = window_size - 1 - i
        coef[i] = coef[mirror_row][::-1]
    _coef_cache[key] = coef
    return coef


def cpp_savitzky_golay_filter(y, hws: int, pno: int) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 1:
        raise ValueError("'y' must contain at least one value.")
    if hws < 1:
        raise ValueError("'hws' must be >= 1.")
    if pno < 0:
        raise ValueError("'pno' must be >= 0.")
    window_size = 2 * hws + 1
    if window_size <= pno:
        raise ValueError("window size must be larger than the polynomial order.")
    if n < window_size:
        raise ValueError("length of 'y' must be >= the full window size.")
    if not np.all(np.isfinite(y)):
        raise ValueError("'y' must contain only finite values.")

    coef = _build_coef(hws, pno)
    out = np.empty(n)

    # Left boundary.
    for i in range(hws):
        out[i] = float(np.dot(coef[i], y[:window_size]))

    # Center: sliding window with the symmetric coefficient row.
    center = coef[hws]
    if n - hws - 1 >= hws:
        # Vectorized sliding dot product.
        windows = np.lib.stride_tricks.sliding_window_view(y, window_size)
        # windows[k] corresponds to y[k:k+window_size]; center index i maps to k=i-hws
        k_lo = 0
        k_hi = (n - hws - 1) - hws  # inclusive
        center_vals = windows[k_lo:k_hi + 1] @ center
        out[hws:n - hws] = center_vals

    # Right boundary.
    start = n - window_size
    for i in range(n - hws, n):
        coef_row = hws + 1 + (i - (n - hws))
        out[i] = float(np.dot(coef[coef_row], y[start:start + window_size]))

    return out
