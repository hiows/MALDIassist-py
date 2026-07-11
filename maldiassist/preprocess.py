"""Port of R/smooth_savitzky_golay.R, R/subtract_baseline.R,
R/preprocess_maldi_spectra.R and R/auxiliary_preprocess.R.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd

from . import baseline as _baseline
from . import savgol as _savgol
from ._util import as_xy, is_single_spectrum, make_df, output_colnames


def smooth_savitzky_golay(data, hws: int = 10, pno: int = 3) -> pd.DataFrame:
    x, y, _ = as_xy(data)
    hws = int(hws)
    pno = int(pno)
    if hws < 1:
        raise ValueError("'hws' must be >= 1.")
    if pno < 0:
        raise ValueError("'pno' must be >= 0.")
    n = y.size
    window_size = 2 * hws + 1
    if window_size <= pno:
        raise ValueError("window size must be larger than 'pno'.")
    if n < window_size:
        raise ValueError("nrow(data) must be >= the full window size.")
    y_smooth = _savgol.cpp_savitzky_golay_filter(y, hws, pno)
    return make_df(x, y_smooth, output_colnames(data))


def subtract_baseline(data, baseline_type: str = "snip", iter_snip: int = 50,
                      hws_tophat: int = 50, nonnegative_baseline: bool = True) -> dict:
    x, y, _ = as_xy(data)
    cols = output_colnames(data)
    if baseline_type == "snip":
        baseline = _baseline.cpp_snip(y, iterations=int(iter_snip),
                                      decreasing=True,
                                      nonnegative=nonnegative_baseline)
    elif baseline_type == "tophat":
        baseline = _baseline.cpp_tophat(y, half_window=int(hws_tophat),
                                        nonnegative=nonnegative_baseline)
    else:
        raise ValueError("baseline_type must be 'snip' or 'tophat'.")
    corrected = np.maximum(y - baseline, 0.0)
    return {
        "raw_data": make_df(x, y, cols),
        "subtracted_data": make_df(x, corrected, cols),
        "baseline": baseline,
        "param": {
            "baseline_type": baseline_type,
            "iter_snip": int(iter_snip),
            "hws_tophat": int(hws_tophat),
            "nonnegative_baseline": nonnegative_baseline,
        },
    }


def _preprocess_single(spectrum, hws_sg, pno_sg, baseline_type, iter_snip, hws_tophat):
    smoothed = smooth_savitzky_golay(spectrum, hws=hws_sg, pno=pno_sg)
    result = subtract_baseline(smoothed, baseline_type=baseline_type,
                               iter_snip=iter_snip, hws_tophat=hws_tophat,
                               nonnegative_baseline=True)
    return result["subtracted_data"]


def preprocess_maldi_spectra(spectra, hws_sg: int = 10, pno_sg: int = 3,
                             baseline_type: str = "snip", iter_snip: int = 50,
                             hws_tophat: int = 50, n_cores: int = 1):
    window_size = 2 * int(hws_sg) + 1
    if pno_sg >= window_size:
        raise ValueError("'pno_sg' must be smaller than the full SG window.")

    if is_single_spectrum(spectra):
        return _preprocess_single(spectra, hws_sg, pno_sg, baseline_type,
                                  iter_snip, hws_tophat)

    if isinstance(spectra, dict):
        out = OrderedDict()
        for name, spec in spectra.items():
            out[name] = _preprocess_single(spec, hws_sg, pno_sg, baseline_type,
                                           iter_snip, hws_tophat)
        return out
    return [
        _preprocess_single(spec, hws_sg, pno_sg, baseline_type, iter_snip, hws_tophat)
        for spec in spectra
    ]
