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


def _normalize_groups(groups, row_labels):
    """Validate ``groups`` and return a list aligned with ``row_labels``.

    Mirrors the R ``heatmap_matched_matrix`` handling: a named mapping
    (``dict`` or a ``pandas.Series`` whose index covers all row labels) is
    reordered by row name, otherwise a positional sequence must match the
    number of rows.
    """
    if groups is None:
        return None

    n = len(row_labels)

    if isinstance(groups, dict):
        missing = [r for r in row_labels if r not in groups]
        if missing:
            raise ValueError(
                "When 'groups' is a mapping, its keys must include all row "
                f"names in 'matched_matrix'. Missing: {missing[:5]}"
            )
        return [groups[r] for r in row_labels]

    if isinstance(groups, pd.Series):
        if all(r in groups.index for r in row_labels):
            return [groups[r] for r in row_labels]
        values = list(groups.to_numpy())
        if len(values) != n:
            raise ValueError(
                "'groups' must have the same length as the number of rows in "
                f"'matched_matrix' ({len(values)} != {n})."
            )
        return values

    values = list(groups)
    if len(values) != n:
        raise ValueError(
            "'groups' must have the same length as the number of rows in "
            f"'matched_matrix' ({len(values)} != {n})."
        )
    return values


def _group_annotation_colors(levels):
    """Map group levels to a fixed Viridis palette (mirrors the R version)."""
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("viridis")
    n = len(levels)
    if n <= 1:
        positions = [0.5]
    else:
        positions = np.linspace(0.0, 1.0, n)
    return {lvl: cmap(pos) for lvl, pos in zip(levels, positions)}


def heatmap_matched_matrix(matched_matrix, row_cluster=True, col_cluster=True,
                           groups=None, title="Matched peaks heatmap",
                           center_at_zero=None, hide_rownames=False,
                           hide_colnames=False, ax=None):
    """Heatmap of a sample-by-marker matched-peak matrix.

    Parameters
    ----------
    matched_matrix : pandas.DataFrame or array-like
        Samples as rows and markers as columns, e.g. the ``detected_matrix``
        or ``delta_mz_matrix`` from :func:`build_matched_matrix`.
    row_cluster, col_cluster : bool
        Whether to hierarchically cluster rows / columns.
    groups : sequence or mapping, optional
        Per-sample group labels. A ``dict`` or named ``pandas.Series`` is
        matched by row name; a plain sequence is used positionally and must
        match the number of rows. When supplied, a Viridis color strip is
        drawn to the left of the heatmap with a matching legend.
    title : str, optional
        Plot title.
    center_at_zero : bool or None
        Color scaling. ``None`` (default) auto-selects: a zero-centered
        diverging palette when the data contains negative values, otherwise a
        sequential palette. A binary 0/1 matrix (such as ``detected_matrix``)
        is drawn with a gray/green detected map. Pass ``True``/``False`` to
        force the behavior.
    hide_rownames, hide_colnames : bool
        Hide sample (row) / marker (column) tick labels.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw the heatmap on. When omitted a new figure is
        created (with a group color strip when ``groups`` is given).

    Returns
    -------
    matplotlib.axes.Axes
        The main heatmap axes.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    from matplotlib.patches import Patch

    if isinstance(matched_matrix, pd.DataFrame):
        row_labels = [str(r) for r in matched_matrix.index]
        col_labels = [str(c) for c in matched_matrix.columns]
        mat = matched_matrix.to_numpy(dtype=float)
    else:
        mat = np.asarray(matched_matrix, dtype=float)
        row_labels = [f"sample_{i+1}" for i in range(mat.shape[0])]
        col_labels = [f"feature_{i+1}" for i in range(mat.shape[1])]

    group_values = _normalize_groups(groups, row_labels)

    if mat.shape[0] < 2:
        row_cluster = False
    if mat.shape[1] < 2:
        col_cluster = False

    row_ord = _cluster_order(mat) if row_cluster else np.arange(mat.shape[0])
    col_ord = _cluster_order(mat.T) if col_cluster else np.arange(mat.shape[1])

    mat_o = mat[np.ix_(row_ord, col_ord)]
    row_labels = [row_labels[i] for i in row_ord]
    col_labels = [col_labels[i] for i in col_ord]
    if group_values is not None:
        group_values = [group_values[i] for i in row_ord]

    vals = mat_o[np.isfinite(mat_o)]
    has_negative = bool(vals.size) and float(np.min(vals)) < 0.0
    is_binary = bool(vals.size) and bool(np.all(np.isin(vals, (0.0, 1.0))))

    if center_at_zero is None:
        center_at_zero = has_negative

    show_colorbar = True
    detected_legend = False
    if center_at_zero:
        max_abs = float(np.max(np.abs(vals))) if vals.size else 1.0
        if max_abs <= 0:
            max_abs = 1.0
        cmap = LinearSegmentedColormap.from_list(
            "bwr_div", ["#2166AC", "#F7F7F7", "#B2182B"])
        vmin, vmax = -max_abs, max_abs
    elif is_binary:
        cmap = ListedColormap(["#F0F0F0", "#1A9850"])
        vmin, vmax = 0.0, 1.0
        show_colorbar = False
        detected_legend = True
    else:
        cmap = plt.get_cmap("viridis").copy()
        vmin = 0.0
        vmax = float(np.max(vals)) if vals.size else None

    cmap.set_bad("#e5e5e5")

    # --- layout: optional group color strip + heatmap ---
    if ax is None:
        fig = plt.figure(figsize=(10, 6))
        if group_values is not None:
            gs = fig.add_gridspec(1, 2, width_ratios=[0.03, 1], wspace=0.02)
            cax = fig.add_subplot(gs[0, 0])
            hax = fig.add_subplot(gs[0, 1])
        else:
            cax = None
            hax = fig.add_subplot(1, 1, 1)
    else:
        hax = ax
        fig = ax.figure
        cax = ax.inset_axes([-0.05, 0.0, 0.03, 1.0]) if group_values is not None else None

    if cax is not None:
        levels = sorted({str(g) for g in group_values})
        colors = _group_annotation_colors(levels)
        level_to_idx = {lvl: i for i, lvl in enumerate(levels)}
        codes = np.array([[level_to_idx[str(g)]] for g in group_values], dtype=float)
        ann_cmap = ListedColormap([colors[lvl] for lvl in levels])
        cax.imshow(codes, aspect="auto", cmap=ann_cmap,
                   vmin=0, vmax=max(len(levels) - 1, 1), interpolation="nearest")
        cax.set_xticks([])
        cax.set_yticks([])
        cax.set_ylabel("samples (grouped)")

    im = hax.imshow(mat_o, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                    interpolation="nearest")
    if title:
        hax.set_title(title)
    if not hide_colnames:
        hax.set_xticks(range(len(col_labels)))
        hax.set_xticklabels(col_labels, rotation=90, fontsize=6)
    else:
        hax.set_xticks([])
    if not hide_rownames:
        hax.set_yticks(range(len(row_labels)))
        hax.set_yticklabels(row_labels, fontsize=6)
    else:
        hax.set_yticks([])

    if detected_legend:
        det_handles = [
            Patch(facecolor="#1A9850", label="detected"),
            Patch(facecolor="#F0F0F0", edgecolor="#cccccc", label="not detected"),
        ]
        leg1 = hax.legend(handles=det_handles, loc="upper right",
                          fontsize=7, framealpha=0.9)
        hax.add_artist(leg1)

    if group_values is not None:
        grp_handles = [Patch(facecolor=colors[lvl], label=lvl) for lvl in levels]
        hax.legend(handles=grp_handles, loc="lower right", fontsize=7,
                   framealpha=0.9, title="group")

    if show_colorbar:
        fig.colorbar(im, ax=hax)

    return hax
