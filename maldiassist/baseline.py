"""Port of src/cpp_baseline_algorithms.cpp (SNIP and TopHat baselines)."""

from __future__ import annotations

import numpy as np


def cpp_snip(intensity, iterations: int = 50, decreasing: bool = True,
             nonnegative: bool = True) -> np.ndarray:
    """SNIP (Sensitive Nonlinear Iterative Peak-clipping) baseline."""
    y = np.asarray(intensity, dtype=float)
    if y.size == 0:
        raise ValueError("'intensity' must not be empty.")
    if not np.all(np.isfinite(y)):
        raise ValueError("'intensity' contains non-finite values.")
    if iterations < 1:
        raise ValueError("'iterations' must be a positive integer.")

    n = y.size
    baseline = y.copy()
    if n < 3:
        if nonnegative:
            baseline[baseline < 0.0] = 0.0
        return baseline

    max_k = min(iterations, (n - 1) // 2)
    ks = range(max_k, 0, -1) if decreasing else range(1, max_k + 1)
    for k in ks:
        previous = baseline
        baseline = previous.copy()
        # centre region update: b_i = min(b_i, (b_{i-k}+b_{i+k})/2)
        cand = 0.5 * (previous[:n - 2 * k] + previous[2 * k:])
        core = previous[k:n - k]
        baseline[k:n - k] = np.minimum(cand, core)
        # boundaries retain previous values (already copied)
    if nonnegative:
        baseline[baseline < 0.0] = 0.0
    return baseline


def _moving_min_centered(x: np.ndarray, half_window: int) -> np.ndarray:
    n = x.size
    out = np.empty(n)
    from collections import deque
    dq = deque()
    right_added = -1
    for i in range(n):
        left = max(0, i - half_window)
        right = min(n - 1, i + half_window)
        while right_added < right:
            right_added += 1
            while dq and x[dq[-1]] >= x[right_added]:
                dq.pop()
            dq.append(right_added)
        while dq and dq[0] < left:
            dq.popleft()
        out[i] = x[dq[0]]
    return out


def _moving_max_centered(x: np.ndarray, half_window: int) -> np.ndarray:
    n = x.size
    out = np.empty(n)
    from collections import deque
    dq = deque()
    right_added = -1
    for i in range(n):
        left = max(0, i - half_window)
        right = min(n - 1, i + half_window)
        while right_added < right:
            right_added += 1
            while dq and x[dq[-1]] <= x[right_added]:
                dq.pop()
            dq.append(right_added)
        while dq and dq[0] < left:
            dq.popleft()
        out[i] = x[dq[0]]
    return out


def cpp_tophat(intensity, half_window: int = 50, nonnegative: bool = True) -> np.ndarray:
    """Morphological Top-Hat baseline: opening = dilation(erosion(x))."""
    y = np.asarray(intensity, dtype=float)
    if y.size == 0:
        raise ValueError("'intensity' must not be empty.")
    if not np.all(np.isfinite(y)):
        raise ValueError("'intensity' contains non-finite values.")
    if half_window < 1:
        raise ValueError("'half_window' must be a positive integer.")
    n = y.size
    hw = min(half_window, max(1, n - 1))
    eroded = _moving_min_centered(y, hw)
    baseline = _moving_max_centered(eroded, hw)
    if nonnegative:
        baseline[baseline < 0.0] = 0.0
    return baseline
