"""Port of the cohort-analysis layer.

R sources: match_peaks.R, find_frequent_mz.R, align_spectra.R,
build_matched_matrix.R, estimate_significance.R.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np
import pandas as pd

from . import peaks as _peaks
from ._util import as_xy
from .rcompat import find_interval, r_hist, r_lowess, r_approx, p_adjust


# ---------------------------------------------------------------------------
# match_peaks
# ---------------------------------------------------------------------------
def match_peaks(peaks, reference_mz, reference_names=None, hws_match: float = 10.0,
                peak_selection_mode: str = "nearest_mz") -> pd.DataFrame:
    reference_mz = np.asarray(reference_mz, dtype=float)
    n_ref = reference_mz.size
    if reference_names is None:
        reference_names = [None] * n_ref
    else:
        reference_names = list(reference_names)

    def _emit(is_matched, detected_mz, detected_intensity):
        delta = detected_mz - reference_mz
        out = pd.DataFrame({
            "reference_name": reference_names,
            "reference_mz": reference_mz,
            "is_matched": is_matched,
            "detected_mz": detected_mz,
            "detected_intensity": detected_intensity,
            "delta_mz": delta,
        })
        out = out.sort_values("reference_mz", kind="mergesort").reset_index(drop=True)
        return out

    mz, intensity, _ = as_xy(peaks)
    if mz.size == 0:
        return _emit(np.zeros(n_ref, dtype=bool),
                     np.full(n_ref, np.nan), np.full(n_ref, np.nan))

    if np.any(np.diff(mz) < 0):
        order = np.argsort(mz, kind="mergesort")
        mz = mz[order]
        intensity = intensity[order]

    # left (0-based start) and right (0-based end, inclusive)
    start0 = find_interval(reference_mz - hws_match, mz, left_open=True)
    right_cnt = find_interval(reference_mz + hws_match, mz)  # 1-based last index
    end0 = right_cnt - 1
    is_matched = start0 <= end0

    detected_mz = np.full(n_ref, np.nan)
    detected_intensity = np.full(n_ref, np.nan)

    for i in np.where(is_matched)[0]:
        lo = int(start0[i])
        hi = int(end0[i])
        idx = np.arange(lo, hi + 1)
        if peak_selection_mode == "nearest_mz":
            j = idx[int(np.argmin(np.abs(mz[idx] - reference_mz[i])))]
        elif peak_selection_mode == "maximum_intensity":
            j = idx[int(np.argmax(intensity[idx]))]
        else:
            raise ValueError("invalid peak_selection_mode")
        detected_mz[i] = mz[j]
        detected_intensity[i] = intensity[j]

    return _emit(is_matched, detected_mz, detected_intensity)


# ---------------------------------------------------------------------------
# find_frequent_mz
# ---------------------------------------------------------------------------
def _r_seq(start, stop, by):
    n = int(math.floor((stop - start) / by + 1e-10))
    return start + by * np.arange(n + 1)


def _pool_peaks(peaks_list):
    if isinstance(peaks_list, dict):
        items = list(peaks_list.values())
    else:
        items = list(peaks_list)
    mzs = []
    ints = []
    for p in items:
        mz, inten, _ = as_xy(p)
        mzs.append(mz)
        ints.append(inten)
    if mzs:
        return np.concatenate(mzs), np.concatenate(ints), len(items)
    return np.zeros(0), np.zeros(0), len(items)


def find_frequent_mz(peaks_list, bin_width: float = 20.0, exclude_mz=None,
                     hws_exclude=None) -> pd.DataFrame:
    if hws_exclude is None:
        hws_exclude = bin_width / 2.0

    cols = ["mz", "median_intensity", "count", "freq_ratio"]

    def empty():
        return pd.DataFrame({c: np.zeros(0) for c in cols})

    if isinstance(peaks_list, dict):
        n_samples = len(peaks_list)
    else:
        n_samples = len(peaks_list)
    if n_samples == 0:
        return empty()

    pooled_mz, pooled_intensity, n_samples = _pool_peaks(peaks_list)
    keep = np.isfinite(pooled_mz) & np.isfinite(pooled_intensity)
    pooled_mz = pooled_mz[keep]
    pooled_intensity = pooled_intensity[keep]
    if pooled_mz.size == 0:
        return empty()

    order = np.argsort(pooled_mz, kind="mergesort")
    pooled_mz = pooled_mz[order]
    pooled_intensity = pooled_intensity[order]

    mz_min = float(pooled_mz.min())
    mz_max = float(pooled_mz.max())
    hws = bin_width / 2.0

    brk_0 = _r_seq(mz_min - bin_width, mz_max + bin_width, bin_width)
    brk_front = _r_seq(mz_min - bin_width - hws, mz_max + bin_width, bin_width)
    brk_back = _r_seq(mz_min - bin_width + hws, mz_max + bin_width, bin_width)

    n_bins = brk_0.size - 1
    if n_bins < 1:
        return empty()

    def _get(arr, idx):
        return arr[idx] if idx < arr.size else np.nan

    rows = []
    seen_mz = set()
    for i in range(n_bins):
        lo_c = [brk_0[i], _get(brk_front, i), _get(brk_back, i)]
        hi_c = [_get(brk_0, i + 1), _get(brk_front, i + 1), _get(brk_back, i + 1)]
        if any(math.isnan(v) for v in lo_c) or any(math.isnan(v) for v in hi_c):
            continue
        win_lo = min(lo_c)
        win_hi = max(hi_c)
        nlt = int(find_interval(win_lo, pooled_mz, left_open=True)[0])
        nle = int(find_interval(win_hi, pooled_mz)[0])
        if nle <= nlt:
            continue
        x = pooled_mz[nlt:nle]
        if np.unique(x).size < 2:
            continue

        mids, density, counts, breaks = r_hist(x, bin_width * 2)
        if mids.size < 2:
            continue
        m = mids
        df = pd.DataFrame({"mid": mids, "dens": density})
        bw = float(np.min(np.diff(mids)))
        if not np.isfinite(bw) or bw <= 0:
            continue

        f = _peaks.get_gauss_kde(df, bw=bw, d=0)
        d1 = _peaks.get_gauss_kde(df, bw=bw, d=1)
        d2 = _peaks.get_gauss_kde(df, bw=bw, d=2)
        ext = _peaks.find_extrema(d1, d2, m)
        x_localmax = ext["local_max"]
        x_localmax = x_localmax[np.isfinite(x_localmax)]
        if x_localmax.size == 0:
            continue
        y_localmax = f(x_localmax)
        xi = float(x_localmax[int(np.argmax(y_localmax))])

        c_lt = int(find_interval(xi - bin_width, pooled_mz, left_open=True)[0])
        c_le = int(find_interval(xi + bin_width, pooled_mz)[0])
        count = c_le - c_lt
        if count > 0:
            sub = pooled_intensity[c_lt:c_le]
        else:
            sub = np.zeros(0)
        sub = sub[np.isfinite(sub) & (sub > 0)]
        if sub.size == 0:
            median_intensity = np.nan
        else:
            median_intensity = 10.0 ** float(np.median(np.log10(sub)))

        if xi in seen_mz:
            continue
        seen_mz.add(xi)
        rows.append((xi, median_intensity, count, count / n_samples))

    if not rows:
        return empty()

    freq_df = pd.DataFrame(rows, columns=cols)

    if exclude_mz is not None:
        exclude_mz = np.asarray(exclude_mz, dtype=float)
        drop = np.zeros(freq_df.shape[0], dtype=bool)
        for e in exclude_mz:
            drop |= np.abs(freq_df["mz"].to_numpy() - e) <= hws_exclude
        freq_df = freq_df.loc[~drop]

    if freq_df.shape[0] == 0:
        return empty()

    freq_df = freq_df.sort_values("mz", kind="mergesort").reset_index(drop=True)
    return freq_df


# ---------------------------------------------------------------------------
# align_spectra
# ---------------------------------------------------------------------------
def _set_first_col(df, values):
    out = df.copy()
    out.iloc[:, 0] = values
    return out


def align_spectra(spectra, peaks_list, bin_width: float = 20.0,
                  alignment_mode: str = "linear", lowess_span: float = 2.0 / 3.0,
                  freq_ratio_cutoff: float = 0.9, hws_alignment: float = 50.0):
    if not isinstance(spectra, dict):
        spectra = OrderedDict((str(i), s) for i, s in enumerate(spectra))
    if not isinstance(peaks_list, dict):
        peaks_list = OrderedDict((str(i), p) for i, p in enumerate(peaks_list))

    common = [nm for nm in spectra.keys() if nm in peaks_list]
    if not common:
        raise ValueError("'spectra' and 'peaks_list' must share sample names.")

    spectra = OrderedDict((nm, spectra[nm]) for nm in common)
    peaks_list = OrderedDict((nm, peaks_list[nm]) for nm in common)

    freq_df = find_frequent_mz(peaks_list, bin_width=bin_width, exclude_mz=None)
    if freq_df.shape[0] == 0:
        raise ValueError("No frequent m/z values found.")
    freq_df = freq_df.loc[freq_df["freq_ratio"] > freq_ratio_cutoff].reset_index(drop=True)
    if freq_df.shape[0] < 2:
        raise ValueError("Fewer than two frequent m/z pass freq_ratio_cutoff.")

    def detected_in_order(matching, std_names):
        # reorder matching (by reference_name) to std_names order
        lookup = {nm: mz for nm, mz in zip(matching["reference_name"], matching["detected_mz"])}
        return np.array([lookup[nm] for nm in std_names], dtype=float)

    def build_matched_mz(detected_by_sample, std_names, is_aligned):
        detected_matrix = np.vstack(detected_by_sample)
        data = {"spectrum_name": common}
        for j, nm in enumerate(std_names):
            data[nm] = detected_matrix[:, j]
        data["is_aligned"] = is_aligned
        return pd.DataFrame(data)

    if alignment_mode == "linear":
        mz = freq_df["mz"].to_numpy()
        intensity = freq_df["median_intensity"].to_numpy()
        min_mz, max_mz = mz.min(), mz.max()
        sel_parts = np.linspace(min_mz, max_mz, 3)
        std_mz = np.empty(2)
        for i in range(2):
            idx = np.where((mz >= sel_parts[i]) & (mz <= sel_parts[i + 1]))[0]
            if idx.size == 0:
                std_mz[i] = np.nan
            else:
                std_mz[i] = mz[idx][int(np.argmax(intensity[idx]))]
        std_names = ["std_1", "std_2"]
        if np.any(np.isnan(std_mz)) or std_mz[0] == std_mz[1]:
            raise ValueError("Could not select two distinct standard anchors.")

        detected_by_sample = []
        for nm in common:
            matching = match_peaks(peaks_list[nm], std_mz, std_names,
                                   hws_match=hws_alignment,
                                   peak_selection_mode="maximum_intensity")
            detected_by_sample.append(detected_in_order(matching, std_names))

        is_aligned = []
        alignment_results = OrderedDict()
        for i, nm in enumerate(common):
            matched_mz = detected_by_sample[i]
            spectrum = spectra[nm]
            pk = peaks_list[nm]
            can = (not np.any(np.isnan(matched_mz))) and (matched_mz[1] != matched_mz[0])
            if can:
                slope = (std_mz[1] - std_mz[0]) / (matched_mz[1] - matched_mz[0])
                intercept = std_mz[1] - slope * matched_mz[1]
                sx, _, _ = as_xy(spectrum)
                px, _, _ = as_xy(pk)
                spectrum = _set_first_col(spectrum, slope * sx + intercept)
                pk = _set_first_col(pk, slope * px + intercept)
                is_aligned.append(True)
            else:
                is_aligned.append(False)
            alignment_results[nm] = {"spectrum": spectrum, "peaks": pk}

        std_mz_named = OrderedDict(zip(std_names, std_mz))
        matched_mz = build_matched_mz(detected_by_sample, std_names, is_aligned)
        return {"alignment_results": alignment_results, "standard_mz": std_mz_named,
                "matched_mz": matched_mz, "alignment_mode": alignment_mode}

    elif alignment_mode == "lowess":
        std_mz = freq_df["mz"].to_numpy()
        std_names = [f"std_{i + 1}" for i in range(std_mz.size)]

        detected_by_sample = []
        for nm in common:
            matching = match_peaks(peaks_list[nm], std_mz, std_names,
                                   hws_match=hws_alignment,
                                   peak_selection_mode="nearest_mz")
            detected_by_sample.append(detected_in_order(matching, std_names))

        is_aligned = []
        alignment_results = OrderedDict()
        for i, nm in enumerate(common):
            matched_mz = detected_by_sample[i]
            spectrum = spectra[nm]
            pk = peaks_list[nm]
            det = matched_mz
            shift = std_mz - matched_mz
            valid = ~np.isnan(det)
            det_v = det[valid]
            shift_v = shift[valid]
            # drop duplicated detected_mz (keep first)
            _, uniq_idx = np.unique(det_v, return_index=True)
            uniq_idx_sorted = np.sort(uniq_idx)
            det_v = det_v[uniq_idx_sorted]
            shift_v = shift_v[uniq_idx_sorted]
            can = det_v.size >= 2
            if can:
                fx, fy = r_lowess(det_v, shift_v, f=lowess_span)
                sx, _, _ = as_xy(spectrum)
                px, _, _ = as_xy(pk)
                spectrum = _set_first_col(spectrum, sx + r_approx(fx, fy, sx, rule=2))
                pk = _set_first_col(pk, px + r_approx(fx, fy, px, rule=2))
                is_aligned.append(True)
            else:
                is_aligned.append(False)
            alignment_results[nm] = {"spectrum": spectrum, "peaks": pk}

        std_mz_named = OrderedDict(zip(std_names, std_mz))
        matched_mz = build_matched_mz(detected_by_sample, std_names, is_aligned)
        return {"alignment_results": alignment_results, "standard_mz": std_mz_named,
                "matched_mz": matched_mz, "alignment_mode": alignment_mode}

    raise ValueError("alignment_mode must be 'linear' or 'lowess'.")


# ---------------------------------------------------------------------------
# build_matched_matrix
# ---------------------------------------------------------------------------
def _r_round_str(x: float) -> str:
    v = round(float(x), 3)
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    if s == "-0":
        s = "0"
    return s


def build_matched_matrix(peaks_list, reference_mz, reference_names=None,
                         hws_match: float = 10.0,
                         peak_selection_mode: str = "nearest_mz"):
    reference_mz = np.asarray(reference_mz, dtype=float)
    n_markers = reference_mz.size

    if reference_names is None:
        marker_cols = [f"mz_{_r_round_str(v)}" for v in reference_mz]
    else:
        marker_cols = list(reference_names)

    if not isinstance(peaks_list, dict):
        peaks_list = OrderedDict((str(i), p) for i, p in enumerate(peaks_list))
    sample_names = list(peaks_list.keys())
    n_samples = len(sample_names)

    matches = OrderedDict()
    for nm in sample_names:
        matches[nm] = match_peaks(peaks_list[nm], reference_mz, reference_names,
                                  hws_match=hws_match,
                                  peak_selection_mode=peak_selection_mode)

    detected = np.zeros((n_samples, n_markers), dtype=int)
    delta = np.full((n_samples, n_markers), np.nan)

    for si, nm in enumerate(sample_names):
        m = matches[nm]
        ref_m = m["reference_mz"].to_numpy()
        det_m = m["detected_mz"].to_numpy()
        del_m = m["delta_mz"].to_numpy()
        # match reference_mz to m$reference_mz (first occurrence)
        idx = _match_first(reference_mz, ref_m)
        detected[si, :] = (~np.isnan(det_m[idx])).astype(int)
        delta[si, :] = del_m[idx]

    keep = detected.sum(axis=0) > 0
    detected = detected[:, keep]
    delta = delta[:, keep]
    reference_mz = reference_mz[keep]
    marker_cols = [c for c, k in zip(marker_cols, keep) if k]
    if reference_names is not None:
        reference_names = [c for c, k in zip(reference_names, keep) if k]

    detected_df = pd.DataFrame(detected, index=sample_names, columns=marker_cols)
    delta_df = pd.DataFrame(delta, index=sample_names, columns=marker_cols)

    return {
        "detected_matrix": detected_df,
        "delta_mz_matrix": delta_df,
        "reference_mz": reference_mz,
        "reference_names": reference_names,
        "sample_names": sample_names,
        "matches": matches,
    }


def _match_first(query, table):
    lookup = {}
    for i, v in enumerate(table):
        if v not in lookup:
            lookup[v] = i
    return np.array([lookup[v] for v in query], dtype=int)


# ---------------------------------------------------------------------------
# estimate_significance
# ---------------------------------------------------------------------------
def estimate_significance(matched_matrix, group, feat_names=None,
                          stat_method: str = "t.test", adj_method: str = "none"):
    from scipy import stats as _stats

    if isinstance(matched_matrix, pd.DataFrame):
        col_names = list(matched_matrix.columns)
        mat = matched_matrix.to_numpy(dtype=float)
    else:
        col_names = None
        mat = np.asarray(matched_matrix, dtype=float)

    n_samples, n_feats = mat.shape
    group = np.asarray(group)
    levels = sorted(set(group.tolist()))
    if len(levels) != 2:
        raise ValueError("'group' must have exactly two distinct levels.")
    is_ref = group == levels[0]

    if feat_names is None:
        feat_names = col_names if col_names is not None else [f"feat_{i+1}" for i in range(n_feats)]

    p = np.empty(n_feats)
    for i in range(n_feats):
        x = mat[is_ref, i]
        y = mat[~is_ref, i]
        if stat_method == "t.test":
            sx = np.std(x, ddof=1) if x.size > 1 else 0.0
            sy = np.std(y, ddof=1) if y.size > 1 else 0.0
            if sx == 0 and sy == 0:
                p[i] = np.nan
                continue
            try:
                p[i] = _stats.ttest_ind(x, y, equal_var=False).pvalue
            except Exception:
                p[i] = np.nan
        elif stat_method == "wilcox":
            try:
                p[i] = _stats.mannwhitneyu(x, y, alternative="two-sided",
                                           use_continuity=True,
                                           method="asymptotic").pvalue
            except Exception:
                p[i] = np.nan
        else:
            raise ValueError("stat_method must be 't.test' or 'wilcox'.")

    adj_p = p_adjust(p, method=adj_method)
    return pd.DataFrame({"feat_names": feat_names, "pvalue": p, "adj_pvalue": adj_p})
