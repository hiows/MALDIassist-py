"""Shared helpers for input coercion and list/dict handling."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd


def as_xy(data):
    """Return (x, y, colnames) from a 2-column DataFrame or ndarray/matrix."""
    if isinstance(data, pd.DataFrame):
        cols = list(data.columns[:2])
        x = np.asarray(data.iloc[:, 0], dtype=float)
        y = np.asarray(data.iloc[:, 1], dtype=float)
        return x, y, cols
    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("data must be a 2-column matrix or data frame.")
    return arr[:, 0].astype(float), arr[:, 1].astype(float), ["x", "y"]


def output_colnames(data):
    """Pick output column names preserving inputs, falling back to ('x','y')."""
    if isinstance(data, pd.DataFrame) and len(data.columns) >= 2:
        cols = list(data.columns[:2])
        if all(isinstance(c, str) and c != "" for c in cols):
            return cols
    return ["x", "y"]


def is_single_spectrum(obj) -> bool:
    return isinstance(obj, (pd.DataFrame, np.ndarray))


def make_df(x, y, colnames):
    df = pd.DataFrame({colnames[0]: np.asarray(x, dtype=float),
                       colnames[1]: np.asarray(y, dtype=float)})
    return df


def iter_named(spectra):
    """Yield (key, value) pairs for a dict/OrderedDict or list of spectra."""
    if isinstance(spectra, dict):
        for k, v in spectra.items():
            yield k, v
    else:
        for i, v in enumerate(spectra):
            yield i, v


def rebuild_like(spectra, results_by_key):
    """Rebuild a dict or list of results mirroring ``spectra``'s structure."""
    if isinstance(spectra, dict):
        out = OrderedDict()
        for k in spectra.keys():
            out[k] = results_by_key[k]
        return out
    return [results_by_key[i] for i in range(len(spectra))]
