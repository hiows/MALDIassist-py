"""K. pneumoniae vs E. coli를 구분하는 top-5 m/z를 찾는 예제.

원본 MALDIassist R 패키지와 동일한 워크플로(로딩 → 전처리 → 피크 검출 → 필터 →
정렬 → 빈발 m/z → 매칭 행렬)로 코호트 특징 행렬을 만든 뒤, 종(species) 라벨을
이용해 두 그룹(E. coli / K. pneumoniae)을 구분하는 m/z 마커를 유의성 검정으로
선정하고 결과 그림을 저장합니다. 결과 그림은 세 가지입니다.

1. 전체 feature 검출 히트맵: `heatmap_matched_matrix()`로 행(샘플)/열(m/z)을
   계층적 군집화해 dendrogram과 종별 색 띠를 함께 그립니다.
2. 유의 feature(adj. p < 0.01)만 추린 검출 히트맵(동일하게 dendrogram 포함).
3. 상위 5개 마커의 종별 검출 빈도 막대그래프.

참고 데이터: 이 예제는 PRIDE 데이터셋 PXD058284 ("Clinical Evaluation of Advanced
MALDI-TOF MS for Carbapenemase Subtyping in Gram-negative Isolates", CC0,
https://www.ebi.ac.uk/pride/archive/projects/PXD058284)를 참고했습니다.

사용법::

    python examples/find_discriminating_mz.py \
        --data-dir TestData \
        --meta metadata.xlsx \
        --out results

주의: TestData(원시 스펙트럼)와 메타(샘플-종 매핑)는 민감 정보이므로 저장소에
포함되지 않습니다. 사용자는 동일한 형식의 Bruker 데이터와 (샘플, 종) 매핑
파일을 준비해 경로만 지정하면 됩니다. 메타 파일은 첫 열이 샘플 이름, 둘째 열이
종 이름인 xlsx/csv면 됩니다.

사전 준비::

    pip install -e ".[viz]"
    pip install openpyxl   # xlsx 메타를 쓸 경우
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import maldiassist as ma

# 구분 대상 두 종
GROUP_A = "Escherichia coli"
GROUP_B = "Klebsiella pneumoniae"
SHORT = {GROUP_A: "E. coli", GROUP_B: "K. pneumoniae"}
COLORS = {GROUP_A: "#2166AC", GROUP_B: "#B2182B"}


def load_species_map(meta_path: str) -> dict:
    """첫 열=샘플 이름, 둘째 열=종 이름인 메타 파일을 dict로 읽는다."""
    ext = os.path.splitext(meta_path)[1].lower()
    if ext in (".xlsx", ".xls"):
        df = pd.read_excel(meta_path)
    else:
        df = pd.read_csv(meta_path)
    sample_col, species_col = df.columns[0], df.columns[1]
    mapping = {}
    for _, row in df.iterrows():
        name = row[sample_col]
        species = row[species_col]
        if pd.isna(name) or pd.isna(species):
            continue
        mapping[str(name).strip()] = str(species).strip()
    return mapping


def run_pipeline(data_dir: str):
    """README와 동일한 파라미터로 코호트 특징 행렬까지 계산한다."""
    raw = ma.load_maldi_spectra(data_dir, return_dir=ROOT)
    pp = ma.preprocess_maldi_spectra(
        raw, hws_sg=10, pno_sg=3, baseline_type="snip", iter_snip=100
    )
    peaks = ma.find_peaks_spectra(
        pp, bw=1.0, hws_peaks=10.0, weight_type="raw",
        cutoff_kappa_peak_strength=0.3,
    )
    fpeaks = ma.filter_peaks_spectra(
        pp, peaks,
        cutoff_peak_intensity=100.0,
        cutoff_peak_prominence=100.0,
        cutoff_peak_strength=0.5,
        normalization_type="raw",
    )
    aligned = ma.align_spectra(
        pp, fpeaks, bin_width=20.0, alignment_mode="linear",
        freq_ratio_cutoff=0.9, hws_alignment=50.0,
    )
    aligned_peaks = {k: v["peaks"] for k, v in aligned["alignment_results"].items()}
    exclude_mz = np.array(list(aligned["standard_mz"].values()))

    freq = ma.find_frequent_mz(aligned_peaks, bin_width=20.0, exclude_mz=exclude_mz)
    matched = ma.build_matched_matrix(
        aligned_peaks, reference_mz=freq["mz"].to_numpy(), hws_match=10.0
    )
    return {
        "detected": matched["detected_matrix"],
        "reference_mz": matched["reference_mz"],
    }


def _marker_mz_map(detected: pd.DataFrame, reference_mz: np.ndarray) -> dict:
    """detected 컬럼명 → reference m/z 값 매핑."""
    return {col: float(mz) for col, mz in zip(detected.columns, reference_mz)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.path.join(ROOT, "TestData"))
    parser.add_argument("--meta", default=os.path.join(ROOT, "metadata.xlsx"))
    parser.add_argument("--out", default=os.path.join(ROOT, "results"))
    parser.add_argument("--topn", type=int, default=5)
    parser.add_argument("--stat", default="wilcox", choices=["wilcox", "t.test"])
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    species_map = load_species_map(args.meta)
    res = run_pipeline(args.data_dir)
    detected = res["detected"]
    mz_map = _marker_mz_map(detected, res["reference_mz"])

    # 두 종에 속하는 샘플만 선택
    # 로더가 중복 처리를 위해 이름에 접두사(예: "TestData/sample")를 붙일 수 있으므로
    # 마지막 토큰을 기준으로 종 라벨을 매칭한다.
    def _norm(name: str) -> str:
        return str(name).replace("\\", "/").split("/")[-1]

    sample_species = {s: species_map.get(_norm(s)) for s in detected.index}
    keep = [s for s in detected.index if sample_species[s] in (GROUP_A, GROUP_B)]
    detected = detected.loc[keep]
    group = np.array([sample_species[s] for s in keep])

    n_a = int(np.sum(group == GROUP_A))
    n_b = int(np.sum(group == GROUP_B))
    print(f"Samples used: {SHORT[GROUP_A]}={n_a}, {SHORT[GROUP_B]}={n_b} "
          f"(total {len(keep)}), markers={detected.shape[1]}")

    # 유의성 검정 (두 그룹 검출 여부 비교)
    sig = ma.estimate_significance(
        detected, group, stat_method=args.stat, adj_method="BH"
    )
    sig = sig.assign(mz=[mz_map[c] for c in sig["feat_names"]])
    sig = sig.sort_values(["adj_pvalue", "pvalue"], kind="mergesort").reset_index(drop=True)
    top = sig.head(args.topn).copy()

    print(f"\nTop-{args.topn} discriminating m/z ({SHORT[GROUP_A]} vs {SHORT[GROUP_B]}):")
    print(top[["mz", "pvalue", "adj_pvalue"]].to_string(index=False))
    top.to_csv(os.path.join(args.out, "top_mz.csv"), index=False)

    top_cols = list(top["feat_names"])
    top_mz = list(top["mz"])

    # 샘플명으로 인덱싱한 종 라벨 (heatmap_matched_matrix의 groups= 인자용)
    group_series = pd.Series(group, index=list(keep), name="group")

    saved = []

    # --- Figure 1: 전체 feature 검출 히트맵 (행/열 계층적 군집화 + 종별 색 띠) ---
    # 제목은 열 dendrogram과 겹치므로 비워 두고 설명은 README 캡션에 맡긴다.
    hax1 = ma.heatmap_matched_matrix(
        detected, groups=group_series,
        hide_rownames=True, hide_colnames=True,
        title="",
    )
    out1 = os.path.join(args.out, "heatmap_all_markers.png")
    hax1.figure.savefig(out1, dpi=150, bbox_inches="tight")
    saved.append(out1)

    # --- Figure 2: 유의 feature(adj. p < 0.01)만 추린 검출 히트맵 ---
    sig_cols = list(sig.loc[sig["adj_pvalue"] < 0.01, "feat_names"])
    if sig_cols:
        hax2 = ma.heatmap_matched_matrix(
            detected[sig_cols], groups=group_series,
            hide_rownames=True, hide_colnames=True,
            title="",
        )
        out2 = os.path.join(args.out, "heatmap_significant.png")
        hax2.figure.savefig(out2, dpi=150, bbox_inches="tight")
        saved.append(out2)
    else:
        print("\n(유의한 feature가 없어 heatmap_significant.png는 생략합니다.)")

    # --- Figure 3: top-5 m/z의 종별 검출 빈도 (집계값) ---
    freq_a = [det_top_frac(detected, group, c, GROUP_A) for c in top_cols]
    freq_b = [det_top_frac(detected, group, c, GROUP_B) for c in top_cols]
    x = np.arange(len(top_mz))
    w = 0.38
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.bar(x - w / 2, freq_a, w, label=f"{SHORT[GROUP_A]} (n={n_a})", color=COLORS[GROUP_A])
    ax3.bar(x + w / 2, freq_b, w, label=f"{SHORT[GROUP_B]} (n={n_b})", color=COLORS[GROUP_B])
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"{m:.1f}" for m in top_mz], rotation=45, ha="right")
    ax3.set_ylabel("Detection frequency")
    ax3.set_ylim(0, 1.05)
    ax3.set_xlabel("m/z marker")
    ax3.set_title(f"Top-{args.topn} discriminating m/z: detection frequency by species")
    ax3.legend()
    out3 = os.path.join(args.out, "top5_detection_frequency.png")
    fig3.savefig(out3, dpi=150, bbox_inches="tight")
    saved.append(out3)

    print("\nSaved figures:")
    for p in saved:
        print(f"  {p}")


def det_top_frac(detected: pd.DataFrame, group: np.ndarray, col: str, species: str) -> float:
    mask = group == species
    if not np.any(mask):
        return 0.0
    return float(np.mean(detected.loc[mask, col].to_numpy(dtype=float)))


if __name__ == "__main__":
    main()
