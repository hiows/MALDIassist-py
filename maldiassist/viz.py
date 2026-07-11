"""Port of R/plot.R and R/heatmap.R to matplotlib.

These are visual (not numeric) outputs, so the goal is functional equivalence:
the same spectrum/peak line plots and the same clustered, zero-centered
matched-matrix heatmap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._util import as_xy


def _restrict(x, y, interest_range):
    if interest_range is None:
        return x, y
    lo, hi = min(interest_range), max(interest_range)
    keep = (x >= lo) & (x <= hi)
    return x[keep], y[keep]


def visualize_spectrum(spectrum, peaks=None, interest_range=None,
                       annotate_topN=False, topN=10, xlim=None, ylim=None,
                       main=None, lwd=1.0, col="black", peaks_lwd=2.0,
                       peaks_col="red", peaks_lty=":", ax=None):
    import matplotlib.pyplot as plt

    x, y, _ = as_xy(spectrum)
    x, y = _restrict(x, y, interest_range)

    if ylim is None:
        lo = min(0.0, float(np.min(y)))
        hi = float(np.max(y))
        offset = (hi - lo) * 0.1
        ylim = (lo, hi + offset)

    if ax is None:
        _, ax = plt.subplots()
    ax.plot(x, y, "-", lw=lwd, color=col)
    ax.set_xlabel("m/z")
    ax.set_ylabel("Intensity")
    if main:
        ax.set_title(main)
    if xlim:
        ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if peaks is not None:
        px, py, _ = as_xy(peaks)
        px, py = _restrict(px, py, interest_range)
        ax.vlines(px, 0, py, lw=peaks_lwd, linestyles=peaks_lty, color=peaks_col)
        if annotate_topN and px.size:
            order = np.argsort(py)[::-1][:topN]
            for xi, yi in zip(px[order], py[order]):
                ax.text(xi, yi, f"{round(float(xi), 2)}", color="blue",
                        fontweight="bold", ha="center", va="bottom")
    return ax


def visualize_spectra(spectra, interest_range=None, xlim=None, ylim=None,
                      main=None, lwd=1.5, cmap="viridis", ax=None):
    import matplotlib.pyplot as plt

    if isinstance(spectra, dict):
        items = list(spectra.values())
    else:
        items = list(spectra)

    processed = []
    pooled_y = []
    for s in items:
        x, y, _ = as_xy(s)
        x, y = _restrict(x, y, interest_range)
        processed.append((x, y))
        pooled_y.append(y)
    pooled_y = np.concatenate(pooled_y) if pooled_y else np.array([0.0])

    if ylim is None:
        lo = min(0.0, float(np.min(pooled_y)))
        hi = float(np.max(pooled_y))
        offset = (hi - lo) * 0.1
        ylim = (lo, hi + offset)

    if ax is None:
        _, ax = plt.subplots()
    cmap_obj = plt.get_cmap(cmap)
    n = len(processed)
    for i, (x, y) in enumerate(processed):
        color = cmap_obj(i / max(n - 1, 1))
        ax.plot(x, y, "-", lw=lwd, color=color)
    ax.set_xlabel("m/z")
    ax.set_ylabel("Intensity")
    if main:
        ax.set_title(main)
    if xlim:
        ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return ax


def _cluster_order(mat):
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import pdist

    filled = np.nan_to_num(mat, nan=0.0)
    if filled.shape[0] < 2:
        return np.arange(filled.shape[0])
    d = pdist(filled, metric="euclidean")
    z = linkage(d, method="complete")
    return leaves_list(z)


def heatmap_matched_matrix(matched_matrix, row_cluster=True, col_cluster=True,
                           groups=None, title="Matched peaks heatmap",
                           center_at_zero=True, hide_rownames=False,
                           hide_colnames=False, ax=None):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    if isinstance(matched_matrix, pd.DataFrame):
        row_labels = [str(r) for r in matched_matrix.index]
        col_labels = [str(c) for c in matched_matrix.columns]
        mat = matched_matrix.to_numpy(dtype=float)
    else:
        mat = np.asarray(matched_matrix, dtype=float)
        row_labels = [f"sample_{i+1}" for i in range(mat.shape[0])]
        col_labels = [f"feature_{i+1}" for i in range(mat.shape[1])]

    if mat.shape[0] < 2:
        row_cluster = False
    if mat.shape[1] < 2:
        col_cluster = False

    row_ord = _cluster_order(mat) if row_cluster else np.arange(mat.shape[0])
    col_ord = _cluster_order(mat.T) if col_cluster else np.arange(mat.shape[1])

    mat_o = mat[np.ix_(row_ord, col_ord)]
    row_labels = [row_labels[i] for i in row_ord]
    col_labels = [col_labels[i] for i in col_ord]

    if center_at_zero:
        vals = mat_o[np.isfinite(mat_o)]
        max_abs = float(np.max(np.abs(vals))) if vals.size else 1.0
        if max_abs <= 0:
            max_abs = 1.0
        cmap = LinearSegmentedColormap.from_list(
            "bwr_div", ["#2166AC", "#F7F7F7", "#B2182B"])
        vmin, vmax = -max_abs, max_abs
    else:
        cmap = plt.get_cmap("viridis")
        vmin = vmax = None

    cmap.set_bad("#e5e5e5")

    if ax is None:
        _, ax = plt.subplots()
    im = ax.imshow(mat_o, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    if title:
        ax.set_title(title)
    if not hide_colnames:
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=90, fontsize=6)
    else:
        ax.set_xticks([])
    if not hide_rownames:
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=6)
    else:
        ax.set_yticks([])
    ax.figure.colorbar(im, ax=ax)
    return ax
