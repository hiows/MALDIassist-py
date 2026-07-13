"""Parity tests: the compiled C++ backend must match the pure-Python golden.

The pure-Python ``_py_*`` implementations in ``maldiassist.spectrum_math`` are
the reference. When the compiled ``_spectrum_math_cpp`` extension is available we
assert that every hot-path function (KDE grid evaluation for derivative orders
0-3, grid root finding, extremum classification) produces the same results, that
NaN positions coincide, and that the full ``find_peaks`` pipeline is identical.

If the extension was not built, the C++ comparisons are skipped (there is nothing
to compare against) but the pure-Python path is still exercised.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from maldiassist import spectrum_math as sm
from maldiassist import peaks as pk

_cpp = sm._cpp
requires_cpp = pytest.mark.skipif(
    _cpp is None,
    reason="compiled _spectrum_math_cpp extension not available",
)

# Strict but not bit-exact: numpy's vectorised exp and C++ scalar std::exp can
# differ by ~1 ULP, so allow a tiny relative tolerance.
RTOL = 1e-9
ATOL = 1e-12

BW = 1.0
CUTOFF = 5.0
TOL = 1e-5
MAX_ITER = 100


def _contig(a):
    return np.ascontiguousarray(a, dtype=float)


def _make_spectrum(n=1200, seed=0):
    """A smooth multi-peak spectrum on a strictly increasing grid."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 300.0, n)
    y = np.zeros_like(x)
    for center, height, width in [
        (40, 100.0, 3.0),
        (95, 60.0, 2.0),
        (150, 140.0, 4.0),
        (210, 45.0, 2.5),
        (260, 80.0, 3.5),
    ]:
        y += height * np.exp(-0.5 * ((x - center) / width) ** 2)
    y += 0.05 * rng.standard_normal(n)
    return _contig(x), _contig(y)


@pytest.fixture(scope="module")
def spectrum():
    return _make_spectrum()


# ---------------------------------------------------------------------------
# gauss_kde_eval / gauss_kde_all_eval
# ---------------------------------------------------------------------------
@requires_cpp
@pytest.mark.parametrize("deriv_order", [0, 1, 2, 3])
def test_gauss_kde_eval_matches(spectrum, deriv_order):
    x, y = spectrum
    ref = sm._py_gauss_kde_eval(x, x, y, BW, deriv_order, True, CUTOFF)
    got = _cpp.gauss_kde_eval(_contig(x), _contig(x), _contig(y), BW,
                              deriv_order, True, CUTOFF)
    np.testing.assert_allclose(got, ref, rtol=RTOL, atol=ATOL)


@requires_cpp
def test_gauss_kde_all_eval_matches(spectrum):
    x, y = spectrum
    ref = sm._py_gauss_kde_all_eval(x, x, y, BW, True, CUTOFF)
    got = _cpp.gauss_kde_all_eval(_contig(x), _contig(x), _contig(y), BW,
                                  True, CUTOFF)
    assert got.shape == ref.shape == (x.size, 4)
    np.testing.assert_allclose(got, ref, rtol=RTOL, atol=ATOL)


@requires_cpp
def test_nan_positions_match_outside_support(spectrum):
    """Grid points far from any observation must yield NaN in both backends."""
    x, y = spectrum
    # Extend the evaluation grid well beyond the observation range so the
    # truncated support is empty at the extremes.
    x_eval = _contig(np.linspace(-200.0, 500.0, 900))
    ref = sm._py_gauss_kde_eval(x_eval, x, y, BW, 0, True, CUTOFF)
    got = _cpp.gauss_kde_eval(x_eval, _contig(x), _contig(y), BW, 0, True, CUTOFF)
    assert np.array_equal(np.isnan(ref), np.isnan(got))
    finite = ~np.isnan(ref)
    np.testing.assert_allclose(got[finite], ref[finite], rtol=RTOL, atol=ATOL)


@requires_cpp
def test_kde_deriv_scalar_matches(spectrum):
    x, y = spectrum
    for xi in (37.3, 95.0, 150.5, 275.9):
        ref = sm._py_kde_deriv_scalar(xi, x, y, BW, 1, True, CUTOFF)
        got = _cpp.kde_deriv_scalar(float(xi), _contig(x), _contig(y), BW, 1,
                                    True, CUTOFF)
        assert np.isclose(got, ref, rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# root finding and extremum classification
# ---------------------------------------------------------------------------
@requires_cpp
def test_find_roots_on_grid_matches(spectrum):
    x, y = spectrum
    d1 = sm._py_gauss_kde_eval(x, x, y, BW, 1, True, CUTOFF)
    ref = sm._py_find_roots_on_grid(x, d1, x, y, BW, 0.0, TOL, MAX_ITER, True, CUTOFF)
    got = _cpp.find_roots_on_grid(_contig(x), _contig(d1), _contig(x), _contig(y),
                                  BW, 0.0, TOL, MAX_ITER, True, CUTOFF)
    assert got.size == ref.size
    np.testing.assert_allclose(np.sort(got), np.sort(ref), rtol=RTOL, atol=1e-6)


@requires_cpp
def test_find_extrema_from_grid_matches(spectrum):
    x, y = spectrum
    d1 = sm._py_gauss_kde_all_eval(x, x, y, BW, True, CUTOFF)[:, 1]
    ref = sm._py_find_extrema_from_grid(x, d1, x, y, BW, TOL, MAX_ITER)
    got = _cpp.find_extrema_from_grid(_contig(x), _contig(d1), _contig(x),
                                      _contig(y), BW, TOL, MAX_ITER, True, CUTOFF)
    for key in ("local_min", "local_max", "plateau"):
        r = np.asarray(ref[key], dtype=float)
        g = np.asarray(got[key], dtype=float)
        assert g.size == r.size, key
        # Both may be the single-NaN sentinel when empty.
        if np.all(np.isnan(r)) and np.all(np.isnan(g)):
            continue
        np.testing.assert_allclose(np.sort(g), np.sort(r), rtol=RTOL, atol=1e-6)


# ---------------------------------------------------------------------------
# full find_peaks pipeline
# ---------------------------------------------------------------------------
def _find_peaks_python(data, **kwargs):
    """Run find_peaks forcing the pure-Python KDE backend via monkeypatch."""
    saved = {
        name: getattr(sm, name)
        for name in ("gauss_kde_eval", "gauss_kde_all_eval", "find_extrema_from_grid")
    }
    try:
        sm.gauss_kde_eval = sm._py_gauss_kde_eval
        sm.gauss_kde_all_eval = sm._py_gauss_kde_all_eval
        sm.find_extrema_from_grid = sm._py_find_extrema_from_grid
        return pk.find_peaks(data, **kwargs)
    finally:
        for name, fn in saved.items():
            setattr(sm, name, fn)


@requires_cpp
def test_find_peaks_pipeline_matches(spectrum):
    x, y = spectrum
    data = pd.DataFrame({"x": x, "y": y})

    ref = _find_peaks_python(data, bw=BW, hws_peaks=10.0)
    got = pk.find_peaks(data, bw=BW, hws_peaks=10.0)  # C++ backed (USE_CPP)

    assert list(got["type"]) == list(ref["type"])
    np.testing.assert_allclose(got.iloc[:, 0].to_numpy(),
                               ref.iloc[:, 0].to_numpy(), rtol=RTOL, atol=1e-6)
    np.testing.assert_allclose(got.iloc[:, 1].to_numpy(),
                               ref.iloc[:, 1].to_numpy(), rtol=RTOL, atol=1e-6)


@requires_cpp
def test_cpp_backend_is_active():
    import os
    disabled = os.environ.get("MALDIASSIST_DISABLE_CPP", "").strip().lower() in {
        "1", "true", "yes", "on",
    }
    if disabled:
        pytest.skip("C++ backend explicitly disabled via MALDIASSIST_DISABLE_CPP")
    assert sm.USE_CPP is True


# ---------------------------------------------------------------------------
# informational benchmark (never fails on timing, only on correctness)
# ---------------------------------------------------------------------------
@requires_cpp
def test_benchmark_speedup(capsys):
    x, y = _make_spectrum(n=4000, seed=1)
    data = pd.DataFrame({"x": x, "y": y})

    ref = _find_peaks_python(data, bw=BW, hws_peaks=10.0)
    got = pk.find_peaks(data, bw=BW, hws_peaks=10.0)
    np.testing.assert_allclose(got.iloc[:, 0].to_numpy(),
                               ref.iloc[:, 0].to_numpy(), rtol=RTOL, atol=1e-6)

    def _time(fn, repeat=3):
        best = float("inf")
        for _ in range(repeat):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        return best

    t_py = _time(lambda: _find_peaks_python(data, bw=BW, hws_peaks=10.0))
    t_cpp = _time(lambda: pk.find_peaks(data, bw=BW, hws_peaks=10.0))
    speedup = t_py / t_cpp if t_cpp else float("inf")
    with capsys.disabled():
        print(f"\n[find_peaks n={x.size}] python={t_py*1e3:.1f} ms  "
              f"cpp={t_cpp*1e3:.1f} ms  speedup={speedup:.1f}x")
