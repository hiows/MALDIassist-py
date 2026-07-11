"""Faithful re-implementations of R base/stats utilities used by MALDIassist.

These helpers reproduce the exact numerical behaviour of the R functions that
the original package relies on, so that the Python port yields identical
results. Implemented here:

- ``r_mad``         : ``stats::mad`` (constant 1.4826, ``median`` type-7-free)
- ``quantile_type7``: ``stats::quantile(type = 7)``
- ``find_interval`` : ``base::findInterval`` (incl. ``left.open``)
- ``r_pretty``      : ``base::pretty`` (Wilkinson-style rounded axis breaks)
- ``r_hist_breaks`` : ``graphics::hist`` breakpoints for a scalar ``breaks``
- ``r_lowess``      : ``stats::lowess`` (Cleveland LOWESS, C ``clowess``)
- ``r_approx``      : ``stats::approx`` linear interpolation with ``rule``
- ``p_adjust``      : ``stats::p.adjust`` (``none``/``BH``/``bonferroni``)
- ``mixedsort``     : ``gtools::mixedsort`` natural ordering
"""

from __future__ import annotations

import math
import re
from typing import List, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# stats::median / stats::mad
# ---------------------------------------------------------------------------
def r_median(x: Sequence[float]) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.median(x))


def r_mad(x: Sequence[float], constant: float = 1.4826, center: float | None = None) -> float:
    """``stats::mad`` with default constant 1.4826.

    ``mad = constant * median(abs(x - center))`` where ``center = median(x)``.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    if center is None:
        center = float(np.median(x))
    return float(constant * np.median(np.abs(x - center)))


# ---------------------------------------------------------------------------
# stats::quantile type 7
# ---------------------------------------------------------------------------
def quantile_type7(x: Sequence[float], prob: float) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    if prob <= 0.0:
        return float(np.min(x))
    if prob >= 1.0:
        return float(np.max(x))
    xs = np.sort(x)
    n = xs.size
    if n == 1:
        return float(xs[0])
    h = 1.0 + (n - 1) * prob
    j = int(math.floor(h))
    gamma = h - j
    xj = xs[j - 1]
    xj1 = xs[min(j, n - 1)]
    return float((1.0 - gamma) * xj + gamma * xj1)


# ---------------------------------------------------------------------------
# base::findInterval
# ---------------------------------------------------------------------------
def find_interval(x, vec, rightmost_closed: bool = False,
                  all_inside: bool = False, left_open: bool = False) -> np.ndarray:
    """Reproduce ``base::findInterval``.

    Returns, for each ``x[i]``, the number of ``vec`` entries that are
    ``<= x[i]`` (or ``< x[i]`` when ``left_open=True``). ``vec`` must be
    sorted (non-decreasing).
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    vec = np.asarray(vec, dtype=float)
    if left_open:
        # count of vec strictly less than x  -> side='left'
        idx = np.searchsorted(vec, x, side="left")
    else:
        # count of vec <= x -> side='right'
        idx = np.searchsorted(vec, x, side="right")
    idx = idx.astype(np.int64)
    if rightmost_closed and vec.size > 0:
        idx[x == vec[-1]] = vec.size - 1
    if all_inside:
        idx = np.clip(idx, 1, max(vec.size - 1, 0))
    return idx


# ---------------------------------------------------------------------------
# base::pretty  (scalar n)  -- port of R's C `R_pretty`
# ---------------------------------------------------------------------------
def _pretty_core(lo: float, up: float, ndiv: int,
                 min_n: int = 1, shrink_sml: float = 0.75,
                 high_u_fact=(1.5, 2.75), eps_correction: int = 0):
    """Port of R's ``R_pretty`` (see src/appl/pretty.c).

    Returns (lo, up, ndiv) after adjusting the endpoints to "nice" numbers and
    the actual number of intervals. The step is ``(up - lo) / ndiv``.
    """
    rounding_eps = 1e-10
    h = high_u_fact[0]
    h5 = high_u_fact[1]

    dx = up - lo
    if dx == 0 and up == 0:
        cell = 1.0
        i_small = True
        u = 1.0
    else:
        cell = max(abs(lo), abs(up))
        # U = 1 + (h5 >= 1.5*h+.5) ? 1/(1+h) : 1.5/(1+h5)
        if h5 >= 1.5 * h + 0.5:
            U = 1.0 / (1.0 + h)
        else:
            U = 1.5 / (1.0 + h5)
        i_small = dx < cell * U * max(1, ndiv) * np.finfo(float).eps * 3

    # OExp / u  scaling
    if i_small:
        if cell > 10:
            cell = 9 + cell / 10.0
        cell *= shrink_sml
        if min_n > 1:
            cell /= min_n
    else:
        cell = dx
        if ndiv > 1:
            cell /= ndiv

    if cell < 20 * np.finfo(float).tiny:
        cell = 20 * np.finfo(float).tiny
    elif cell * 10 > np.finfo(float).max:
        cell = np.finfo(float).max / 10.0

    base = 10.0 ** math.floor(math.log10(cell))
    unit = base
    if (2 * base) - cell < h * (cell - unit):
        unit = 2.0 * base
        if (5 * base) - cell < h5 * (cell - unit):
            unit = 5.0 * base
            if (10 * base) - cell < h * (cell - unit):
                unit = 10.0 * base

    # find_base
    ns = math.floor(lo / unit + rounding_eps)
    nu = math.ceil(up / unit - rounding_eps)

    if eps_correction and (eps_correction > 1 or not i_small):
        if lo != 0.0:
            lo *= (1 - np.finfo(float).eps)
        else:
            lo = -np.finfo(float).tiny
        if up != 0.0:
            up *= (1 + np.finfo(float).eps)
        else:
            up = +np.finfo(float).tiny

    while ns * unit > lo + rounding_eps * unit:
        ns -= 1
    while nu * unit < up - rounding_eps * unit:
        nu += 1

    k = int(0.5 + nu - ns)
    if k < min_n:
        k_add = min_n - k
        if ns >= 0.0:
            nu += k_add / 2
            ns -= k_add / 2 + k_add % 2
        else:
            ns -= k_add / 2
            nu += k_add / 2 + k_add % 2
        ndiv = min_n
    else:
        ndiv = k

    lo_out = ns * unit
    up_out = nu * unit
    return lo_out, up_out, int(ndiv)


def r_pretty(x, n: int = 5, min_n=None, shrink_sml: float = 0.75,
             high_u_fact=(1.5, 2.75)) -> np.ndarray:
    """Reproduce ``base::pretty`` for a numeric vector ``x``."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    lo = float(np.min(x))
    up = float(np.max(x))
    if min_n is None:
        min_n = n // 3
    lo_o, up_o, ndiv = _pretty_core(lo, up, n, min_n=min_n,
                                    shrink_sml=shrink_sml,
                                    high_u_fact=high_u_fact,
                                    eps_correction=0)
    step = (up_o - lo_o) / ndiv
    breaks = lo_o + step * np.arange(ndiv + 1)
    return breaks


# ---------------------------------------------------------------------------
# graphics::hist breakpoints for a scalar `breaks`
# ---------------------------------------------------------------------------
def r_hist_breaks(x, breaks_scalar) -> np.ndarray:
    """Compute ``graphics::hist(x, breaks = breaks_scalar)`` breakpoints.

    R: when ``breaks`` is a single number, it is used as a *suggestion* and the
    actual breaks are ``pretty(range(x), n = breaks, min.n = 1)``, then the
    range is extended if data falls exactly on the outer edges.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    rng = (float(np.min(x)), float(np.max(x)))
    breaks = r_pretty(np.array(rng), n=int(breaks_scalar), min_n=1)
    return breaks


def r_hist(x, breaks_scalar):
    """Minimal ``graphics::hist`` replica returning (mids, density, counts, breaks).

    Reproduces the default ``right = TRUE`` (right-closed) binning and the
    ``include.lowest`` handling used by R when ``breaks`` covers the data.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    breaks = r_hist_breaks(x, breaks_scalar)
    nb = breaks.size
    # right-closed intervals (a, b]; leftmost includes lower endpoint.
    # counts[k] = #{ breaks[k] < x <= breaks[k+1] }, with x==breaks[0] in bin 0
    idx = np.searchsorted(breaks, x, side="left")  # position among breaks
    # x equal to a break belongs to the bin on its left (right-closed),
    # except x == breaks[0] which goes to bin 0.
    counts = np.zeros(nb - 1, dtype=float)
    for xi in x:
        # find bin: largest k such that breaks[k] < xi <= breaks[k+1]
        k = int(np.searchsorted(breaks, xi, side="left")) - 1
        if xi == breaks[0]:
            k = 0
        if k < 0:
            k = 0
        if k > nb - 2:
            k = nb - 2
        counts[k] += 1.0
    diffs = np.diff(breaks)
    n = x.size
    density = counts / (n * diffs)
    mids = 0.5 * (breaks[:-1] + breaks[1:])
    return mids, density, counts, breaks


# ---------------------------------------------------------------------------
# stats::approx (linear), rule = 1 or 2
# ---------------------------------------------------------------------------
def r_approx(x, y, xout, rule: int = 2, ties="mean") -> np.ndarray:
    """Linear interpolation reproducing ``stats::approx`` with ``method="linear"``.

    ``rule = 2`` uses the closest data extreme for points outside the range.
    Duplicate ``x`` values are collapsed by ``ties`` (default mean).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x, kind="mergesort")
    x = x[order]
    y = y[order]
    # collapse ties
    if np.any(np.diff(x) == 0):
        ux, inv = np.unique(x, return_inverse=True)
        uy = np.zeros_like(ux)
        for i in range(ux.size):
            vals = y[inv == i]
            if ties == "mean":
                uy[i] = np.mean(vals)
            else:
                uy[i] = vals[0]
        x, y = ux, uy
    xout = np.atleast_1d(np.asarray(xout, dtype=float))
    left = y[0] if rule == 2 else np.nan
    right = y[-1] if rule == 2 else np.nan
    out = np.interp(xout, x, y, left=left, right=right)
    return out


# ---------------------------------------------------------------------------
# stats::lowess  (Cleveland LOWESS, port of C `clowess`)
# ---------------------------------------------------------------------------
def _lowest(x, y, xs, nleft, nright, w, userw, rw):
    n = x.size
    range_ = x[n - 1] - x[0]
    h = max(xs - x[nleft], x[nright] - xs)
    h9 = 0.999 * h
    h1 = 0.001 * h

    a = 0.0
    j = nleft
    nrt = nleft
    while j < n:
        w[j] = 0.0
        r = abs(x[j] - xs)
        if r <= h9:
            if r <= h1:
                w[j] = 1.0
            else:
                w[j] = (1.0 - (r / h) ** 3) ** 3
            if userw:
                w[j] *= rw[j]
            a += w[j]
        elif x[j] > xs:
            break
        j += 1
        nrt = j - 1

    nrt = j - 1  # rightmost pt (may be greater than nright because of ties)
    if a <= 0.0:
        return False, 0.0
    # normalize
    w[nleft:nrt + 1] /= a
    if h > 0.0:
        a = 0.0
        for k in range(nleft, nrt + 1):
            a += w[k] * x[k]
        b = xs - a
        c = 0.0
        for k in range(nleft, nrt + 1):
            c += w[k] * (x[k] - a) ** 2
        if math.sqrt(c) > 0.001 * range_:
            b /= c
            for k in range(nleft, nrt + 1):
                w[k] *= (b * (x[k] - a) + 1.0)
    ys = 0.0
    for k in range(nleft, nrt + 1):
        ys += w[k] * y[k]
    return True, ys


def r_lowess(x, y, f: float = 2.0 / 3.0, iter: int = 3, delta: float | None = None):
    """Port of ``stats::lowess`` (C ``clowess`` in src/library/stats/src/lowess.c).

    Returns sorted ``(xs, ys)`` matching R's output. ``x`` is sorted ascending
    (with ``y`` following the same permutation), as R does internally.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x, kind="mergesort")
    x = x[order].copy()
    y = y[order].copy()
    n = x.size
    ys = np.zeros(n)
    if n < 2:
        return x, y.copy()
    if delta is None:
        delta = 0.01 * (x[n - 1] - x[0])

    rw = np.ones(n)     # robustness weights
    res = np.zeros(n)   # residuals
    w = np.zeros(n)     # scratch fit weights

    ns = max(min(int(f * n + 1e-7), n), 2)
    for iteration in range(1, iter + 2):
        nleft = 0
        nright = ns - 1
        last = -1
        i = 0
        while True:
            while nright < n - 1:
                d1 = x[i] - x[nleft]
                d2 = x[nright + 1] - x[i]
                if d1 <= d2:
                    break
                nleft += 1
                nright += 1
            ok, ys_i = _lowest(x, y, x[i], nleft, nright, w,
                               iteration > 1, rw)
            ys[i] = ys_i if ok else y[i]
            if last < i - 1:
                denom = x[i] - x[last]
                for j in range(last + 1, i):
                    alpha = (x[j] - x[last]) / denom
                    ys[j] = alpha * ys[i] + (1.0 - alpha) * ys[last]
            last = i
            cut = x[last] + delta
            i = last + 1
            while i < n:
                if x[i] > cut:
                    break
                if x[i] == x[last]:
                    ys[i] = ys[last]
                    last = i
                i += 1
            i = max(last + 1, i - 1)
            if last >= n - 1:
                break
        for k in range(n):
            res[k] = y[k] - ys[k]
        if iteration > iter:
            break
        # robustness weights: cmad = 6 * median(|res|)
        abs_res = np.abs(res)
        rw_sorted = np.sort(abs_res)
        m1 = n // 2
        if n % 2 == 0:
            m2 = n - m1 - 1
            cmad = 3.0 * (rw_sorted[m1] + rw_sorted[m2])
        else:
            cmad = 6.0 * rw_sorted[m1]
        c9 = 0.999 * cmad
        c1 = 0.001 * cmad
        for k in range(n):
            r = abs_res[k]
            if cmad == 0.0:
                rw[k] = 1.0
            elif r <= c1:
                rw[k] = 1.0
            elif r <= c9:
                rw[k] = (1.0 - (r / cmad) ** 2) ** 2
            else:
                rw[k] = 0.0
    return x, ys


# ---------------------------------------------------------------------------
# stats::p.adjust
# ---------------------------------------------------------------------------
def p_adjust(p, method: str = "none") -> np.ndarray:
    p = np.asarray(p, dtype=float)
    n = np.sum(np.isfinite(p))
    out = np.full(p.shape, np.nan)
    finite = np.isfinite(p)
    if method == "none":
        return p.copy()
    if method == "bonferroni":
        out[finite] = np.minimum(p[finite] * n, 1.0)
        return out
    if method == "BH":
        pf = p[finite]
        m = pf.size
        order = np.argsort(pf)[::-1]  # decreasing
        ranks = np.arange(m, 0, -1)  # m, m-1, ..., 1 aligned to decreasing order
        adj = np.minimum.accumulate(np.minimum(m / ranks * pf[order], 1.0))
        result = np.empty(m)
        result[order] = adj
        out[finite] = result
        return out
    raise ValueError(f"Unsupported p.adjust method: {method}")


# ---------------------------------------------------------------------------
# gtools::mixedsort natural ordering
# ---------------------------------------------------------------------------
_num_re = re.compile(r"(\d+\.?\d*(?:[eE][+-]?\d+)?)")


def _mixed_key(s: str):
    parts = _num_re.split(str(s))
    key = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            try:
                key.append((1, float(part)))
            except ValueError:
                key.append((0, part))
        else:
            if part != "":
                key.append((0, part))
    return key


def mixedsort(values: Sequence[str]) -> List[str]:
    return sorted(values, key=_mixed_key)


def mixedorder(values: Sequence[str]) -> List[int]:
    idx = list(range(len(values)))
    idx.sort(key=lambda i: _mixed_key(values[i]))
    return idx
