"""maldiassist: Python port of the MALDIassist R package.

Mathematical utilities for MALDI-TOF mass spectrometry: Bruker loading,
smoothing and baseline correction, Gaussian kernel-regression peak detection
(including shoulder peaks), peak-quality metrics and filtering, and cohort
feature analysis.
"""

from __future__ import annotations

from .io import load_maldi_spectra
from .preprocess import (
    preprocess_maldi_spectra,
    smooth_savitzky_golay,
    subtract_baseline,
)
from .peaks import (
    build_kde_spectra,
    build_kde_spectrum,
    estimate_peak_boundaries,
    find_extrema,
    find_extrema_fast,
    find_peaks,
    find_peaks_fast,
    find_peaks_spectra,
    find_peaks_spectra_fast,
    get_curvature_fun,
    get_gauss_kde,
)
from .metrics import calculate_peak_prominence, estimate_peak_strength
from .filter import filter_peaks, filter_peaks_spectra
from .cohort import (
    align_spectra,
    build_matched_matrix,
    estimate_significance,
    find_frequent_mz,
    match_peaks,
)

try:  # optional (matplotlib)
    from .viz import heatmap_matched_matrix, visualize_spectra, visualize_spectrum
except Exception:  # pragma: no cover - matplotlib optional
    pass

__version__ = "0.2.2"

__all__ = [
    "load_maldi_spectra",
    "preprocess_maldi_spectra",
    "smooth_savitzky_golay",
    "subtract_baseline",
    "get_gauss_kde",
    "build_kde_spectrum",
    "build_kde_spectra",
    "get_curvature_fun",
    "find_extrema",
    "find_extrema_fast",
    "find_peaks",
    "find_peaks_fast",
    "find_peaks_spectra",
    "find_peaks_spectra_fast",
    "estimate_peak_boundaries",
    "calculate_peak_prominence",
    "estimate_peak_strength",
    "filter_peaks",
    "filter_peaks_spectra",
    "match_peaks",
    "find_frequent_mz",
    "align_spectra",
    "build_matched_matrix",
    "estimate_significance",
    "visualize_spectrum",
    "visualize_spectra",
    "heatmap_matched_matrix",
]
