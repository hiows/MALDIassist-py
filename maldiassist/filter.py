"""Port of R/filter_peaks.R."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd

from ._util import as_xy, is_single_spectrum
from .metrics import calculate_peak_prominence, estimate_peak_strength
from .rcompat import r_mad


def filter_peaks(data, peaks, cutoff_peak_intensity=None,
                 cutoff_peak_prominence=None, cutoff_peak_strength=0.2,
                 k: float = 1.0, normalization_type: str = "raw"):
    x, y, _ = as_xy(data)
    if isinstance(peaks, pd.DataFrame):
        n_peaks = peaks.shape[0]
    else:
        n_peaks = np.asarray(peaks).shape[0]
    if n_peaks == 0:
        return peaks

    px, intensity, _ = as_xy(peaks)

    if cutoff_peak_intensity is None:
        cutoff_peak_intensity = r_mad(y)
    if cutoff_peak_prominence is None:
        cutoff_peak_prominence = r_mad(y)

    for val, name in ((cutoff_peak_intensity, "cutoff_peak_intensity"),
                      (cutoff_peak_prominence, "cutoff_peak_prominence"),
                      (cutoff_peak_strength, "cutoff_peak_strength")):
        if not np.isfinite(val) or val < 0:
            raise ValueError(f"'{name}' must be a non-negative finite scalar.")
    if not np.isfinite(k) or k <= 0:
        raise ValueError("'k' must be a positive finite scalar.")

    prominence = calculate_peak_prominence(data, peaks, valley_type="lower")
    strength = estimate_peak_strength(data, peaks, k=k,
                                      normalization_type=normalization_type)

    keep = (np.isfinite(intensity) & np.isfinite(prominence) & np.isfinite(strength)
            & (intensity > cutoff_peak_intensity)
            & (prominence > cutoff_peak_prominence)
            & (strength > cutoff_peak_strength))

    if isinstance(peaks, pd.DataFrame):
        out = peaks.loc[keep].reset_index(drop=True)
        return out
    arr = np.asarray(peaks)
    return arr[keep]


def filter_peaks_spectra(spectra, peaks_list, cutoff_peak_intensity=None,
                         cutoff_peak_prominence=None, cutoff_peak_strength=0.2,
                         k: float = 1.0, normalization_type: str = "raw",
                         n_cores: int = 1):
    kwargs = dict(cutoff_peak_intensity=cutoff_peak_intensity,
                  cutoff_peak_prominence=cutoff_peak_prominence,
                  cutoff_peak_strength=cutoff_peak_strength, k=k,
                  normalization_type=normalization_type)
    if is_single_spectrum(spectra):
        return filter_peaks(spectra, peaks_list, **kwargs)

    if isinstance(spectra, dict) and isinstance(peaks_list, dict):
        common = [nm for nm in spectra.keys() if nm in peaks_list]
        return OrderedDict(
            (nm, filter_peaks(spectra[nm], peaks_list[nm], **kwargs)) for nm in common
        )
    # positional lists
    return [filter_peaks(s, p, **kwargs) for s, p in zip(spectra, peaks_list)]
