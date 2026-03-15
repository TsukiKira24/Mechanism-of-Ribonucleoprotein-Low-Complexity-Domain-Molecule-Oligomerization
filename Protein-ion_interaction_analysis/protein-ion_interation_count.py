#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
protein-ion_interation_count.py
================================
Compare protein–ion interaction frequencies across pH conditions by
combining data from FOLD1 (monomer) and FOLD2 (dimer) trajectories.

The script reads the per-frame ion-count files produced by
``ion_interactions_per_frame.py`` (tab-delimited ``frame  ion_counts``),
pools them across folds, and produces:

    * Violin plots with per-frame strip points, colour-coded by fold,
      and Mann–Whitney U significance brackets.
    * Kruskal–Wallis and pairwise Mann–Whitney U statistics.
    * Cl⁻ / Na⁺ ratio analysis per frame.
    * **CSV exports** of the combined dataset, summary statistics, pairwise
      test results, and ratio table for repository deposition.

Inputs
------
For each fold (1, 2) × pH (40, 74, 85) × ion (Cl⁻, Na⁺)::

    FOLD<fold>_pH<PH>_Frame_Analysis_<ion>.txt

Outputs
-------
    * ``Combined_FOLD1_FOLD2_*_violin_plot.png``
    * ``combined_folds_frame_data.csv``
    * ``combined_folds_summary_statistics.csv``
    * ``combined_folds_pairwise_tests.csv``
    * ``combined_folds_cl_na_ratios.csv``

Usage
-----
    python protein-ion_interation_count.py
    
Requirements
------------
    NumPy, pandas, matplotlib, seaborn, scipy

Author  : Aleksandra Wosztyl
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kruskal, mannwhitneyu


# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════

PH_CONDITIONS: List[Dict[str, str]] = [
    {"PH": "40", "PHdot": "4.0"},
    {"PH": "74", "PHdot": "7.4"},
    {"PH": "85", "PHdot": "8.5"},
]

# Mapping from filename pH codes to float values
_PH_MAP: Dict[str, float] = {"40": 4.0, "74": 7.4, "85": 8.5}

# Folds to combine
FOLDS: List[str] = ["1", "2"]

# Ion types expected in the frame-analysis files
ION_TYPES: List[str] = ["Cl-", "Na+"]


# ═══════════════════════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════════════════════


def load_frame_data(
    fold: str, ph: str, ion_type: str
) -> pd.DataFrame:
    """Load a single tab-delimited frame-analysis file.

    Returns a DataFrame with added ``fold``, ``pH``, and ``ion_type``
    columns, or an empty DataFrame if the file is missing.
    """
    filename = f"FOLD{fold}_pH{ph}_Frame_Analysis_{ion_type}.txt"

    if not os.path.exists(filename):
        print(f"  Warning: {filename} not found")
        return pd.DataFrame()

    try:
        df = pd.read_csv(filename, sep="\t")
        df["fold"] = fold
        df["pH"] = _PH_MAP.get(ph, float(ph))
        df["ion_type"] = ion_type
        return df
    except Exception as exc:
        print(f"  Error loading {filename}: {exc}")
        return pd.DataFrame()


def combine_all_data() -> pd.DataFrame:
    """Load and concatenate data for every fold × pH × ion combination."""
    frames: List[pd.DataFrame] = []

    for cond in PH_CONDITIONS:
        ph = cond["PH"]
        ph_dot = cond["PHdot"]
        print(f"Loading data for pH {ph_dot} …")

        for fold in FOLDS:
            for ion in ION_TYPES:
                df = load_frame_data(fold, ph, ion)
                if not df.empty:
                    frames.append(df)
                    print(f"  FOLD{fold} {ion}: {len(df)} frames")

    if not frames:
        print("No data loaded!")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nCombined dataset: {len(combined)} data points")
    return combined


# ═══════════════════════════════════════════════════════════════════════════
#  Statistical analysis
# ═══════════════════════════════════════════════════════════════════════════


def perform_statistical_analysis(
    df: pd.DataFrame, ion_type: str
) -> Dict[Tuple[float, float], float]:
    """Kruskal–Wallis + pairwise Mann–Whitney U for one ion type.

    Returns
    -------
    dict
        ``{(pH_1, pH_2): p_value}`` for each pairwise comparison.
    """
    print(f"\nStatistical Analysis for {ion_type}:")
    print("=" * 50)

    ion_data = df[df["ion_type"] == ion_type]
    if ion_data.empty:
        print(f"  No data for {ion_type}")
        return {}

    ph_vals = sorted(ion_data["pH"].unique())

    # Kruskal–Wallis omnibus test
    groups = [
        ion_data.loc[ion_data["pH"] == ph, "ion_counts"].values
        for ph in ph_vals
    ]
    h_stat, h_p = kruskal(*groups)
    print(f"  Kruskal–Wallis: H = {h_stat:.3f}, p = {h_p:.4f}")

    # Pairwise Mann–Whitney U
    pairwise: Dict[Tuple[float, float], float] = {}
    for i, ph1 in enumerate(ph_vals):
        for ph2 in ph_vals[i + 1:]:
            g1 = ion_data.loc[ion_data["pH"] == ph1, "ion_counts"]
            g2 = ion_data.loc[ion_data["pH"] == ph2, "ion_counts"]

            if len(g1) == 0 or len(g2) == 0:
                continue

            u_stat, p_val = mannwhitneyu(g1, g2, alternative="two-sided")
            pairwise[(ph1, ph2)] = p_val

            # Rank-biserial effect size
            n1, n2 = len(g1), len(g2)
            r = abs(1 - (2 * u_stat) / (n1 * n2))

            if r < 0.1:
                mag = "negligible"
            elif r < 0.3:
                mag = "small"
            elif r < 0.5:
                mag = "medium"
            else:
                mag = "large"

            print(f"  pH {ph1} vs {ph2}: "
                  f"p = {p_val:.4f}, r = {r:.3f} ({mag})")

    return pairwise


# ═══════════════════════════════════════════════════════════════════════════
#  Visualisation
# ═══════════════════════════════════════════════════════════════════════════


def _add_stat_annotation(
    ax, x1: int, x2: int, y: float, p_value: float,
    height_offset: float = 0.1,
) -> None:
    """Draw a significance bracket between positions *x1* and *x2*."""
    if p_value < 0.001:
        sym = "***"
    elif p_value < 0.01:
        sym = "**"
    elif p_value < 0.05:
        sym = "*"
    else:
        sym = "ns"

    ax.plot(
        [x1, x1, x2, x2],
        [y, y + height_offset, y + height_offset, y],
        lw=1, c="black",
    )
    ax.text(
        (x1 + x2) * 0.5, y + height_offset, sym,
        ha="center", va="bottom", fontsize=12, fontweight="bold",
    )


def create_violin_plot(
    df: pd.DataFrame,
    ion_type: str,
    pairwise_results: Dict[Tuple[float, float], float],
) -> None:
    """Violin plot with per-frame dots coloured by fold."""
    plot_data = df[df["ion_type"] == ion_type].copy()
    if plot_data.empty:
        print(f"  No data to plot for {ion_type}")
        return

    plt.figure(figsize=(8, 10))
    sns.set_style("ticks")

    # pH-inspired colour palette for the violins
    ph_colors: Dict[float, str] = {
        4.0: "#bfff00",    # lime green  — acidic
        7.4: "#228B22",    # forest green — neutral
        8.5: "#008B8B",    # dark cyan   — alkaline
    }
    ph_vals = sorted(plot_data["pH"].unique())
    palette = [ph_colors[ph] for ph in ph_vals]

    ax = sns.violinplot(
        data=plot_data, x="pH", y="ion_counts",
        palette=palette, alpha=0.7, inner=None,
    )

    # Overlay individual frames, coloured by fold
    fold_colors = {"1": "#FF6347", "2": "#4169E1"}  # tomato / royal blue
    for fold in FOLDS:
        subset = plot_data[plot_data["fold"] == fold]
        if not subset.empty:
            sns.stripplot(
                data=subset, x="pH", y="ion_counts",
                size=3, alpha=0.6, jitter=0.3,
                color=fold_colors[fold], ax=ax,
                label=f"FOLD{fold}",
            )

    # Labels and title
    ion_sym = ion_type.replace("-", "⁻").replace("+", "⁺")
    ax.set_ylabel(f"{ion_sym} Interactions per Frame", fontsize=16)
    ax.set_xlabel("")
    ax.set_title(
        f"Protein–{ion_sym} Interactions per Frame by pH\n"
        f"(Combined FOLD1 and FOLD2)",
        fontsize=14, pad=20,
    )

    y_max = plot_data["ion_counts"].max()
    ax.set_ylim(-0.5, y_max * 1.3)

    # Significance brackets
    if pairwise_results:
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        base = y_max + y_range * 0.05
        ph_pos = {ph: i for i, ph in enumerate(ph_vals)}
        n = 0
        for (ph1, ph2), p_val in pairwise_results.items():
            if ph1 in ph_pos and ph2 in ph_pos:
                _add_stat_annotation(
                    ax, ph_pos[ph1], ph_pos[ph2],
                    base + n * y_range * 0.06,
                    p_val, height_offset=y_range * 0.02,
                )
                n += 1

    ax.set_xticklabels([f"pH {ph}" for ph in ph_vals], fontsize=14)
    ax.tick_params(axis="y", labelsize=12)

    # Fold legend
    legend_handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=fold_colors[f], markersize=8,
            alpha=0.6, label=f"FOLD{f}",
        )
        for f in FOLDS
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=10)

    ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout()

    # Save
    ion_tag = ion_type.replace("-", "Cl").replace("+", "Na")
    fname = f"Combined_FOLD1_FOLD2_{ion_tag}_violin_plot.png"
    plt.savefig(
        fname, dpi=300, bbox_inches="tight",
        facecolor="white", edgecolor="none",
    )
    plt.show()
    print(f"  Plot saved → {fname}")


# ═══════════════════════════════════════════════════════════════════════════
#  Summary statistics (console + CSV)
# ═══════════════════════════════════════════════════════════════════════════


def _indent(text: str, n: int = 4) -> str:
    """Prepend *n* spaces to every line — portable across pandas versions."""
    pad = " " * n
    return "\n".join(pad + line for line in text.splitlines())


def generate_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Print and return grouped summary statistics."""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE SUMMARY STATISTICS")
    print("=" * 60)

    all_stats: List[pd.DataFrame] = []

    for ion in ION_TYPES:
        ion_data = df[df["ion_type"] == ion]
        if ion_data.empty:
            print(f"\n  No data for {ion}")
            continue

        print(f"\n{ion} Interactions:")
        print("-" * 40)
        print(f"  Total frames: {len(ion_data)}")
        print(f"  Mean: {ion_data['ion_counts'].mean():.2f}")
        print(f"  Median: {ion_data['ion_counts'].median():.2f}")
        print(f"  Std: {ion_data['ion_counts'].std():.2f}")
        print(f"  Range: {ion_data['ion_counts'].min()}"
              f"–{ion_data['ion_counts'].max()}")

        # By pH
        print("\n  By pH:")
        by_ph = ion_data.groupby("pH")["ion_counts"].agg(
            ["count", "mean", "median", "std", "min", "max"]
        ).round(3)
        print(_indent(by_ph.to_string(), 4))

        # By fold
        print("\n  By FOLD:")
        by_fold = ion_data.groupby("fold")["ion_counts"].agg(
            ["count", "mean", "median", "std", "min", "max"]
        ).round(3)
        print(_indent(by_fold.to_string(), 4))

        # By pH × fold
        print("\n  By pH × FOLD:")
        by_both = ion_data.groupby(["pH", "fold"])["ion_counts"].agg(
            ["count", "mean", "median", "std"]
        ).round(3)
        print(_indent(by_both.to_string(), 4))

        # Collect for CSV export
        by_both_flat = by_both.reset_index()
        by_both_flat["ion_type"] = ion
        all_stats.append(by_both_flat)

    if all_stats:
        return pd.concat(all_stats, ignore_index=True)
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════
#  Cl⁻ / Na⁺ ratio analysis
# ═══════════════════════════════════════════════════════════════════════════


def compute_cl_na_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-frame Cl⁻ / Na⁺ ratio across all folds and pH values.

    Returns a DataFrame with columns: ``fold, pH, frame, Cl-, Na+,
    Cl_Na_ratio``.  Infinite ratios (Na⁺ = 0, Cl⁻ > 0) are replaced
    with ``NaN`` so downstream stats are not distorted.
    """
    print(f"\n{'=' * 60}")
    print("Cl⁻ / Na⁺ RATIO ANALYSIS")
    print(f"{'=' * 60}")

    pivot = df.pivot_table(
        index=["fold", "pH", "frame"],
        columns="ion_type",
        values="ion_counts",
        fill_value=0,
    ).reset_index()

    # Compute ratio; replace inf with NaN for cleaner downstream use
    pivot["Cl_Na_ratio"] = np.where(
        pivot["Na+"] > 0,
        pivot["Cl-"] / pivot["Na+"],
        np.where(pivot["Cl-"] > 0, np.nan, 0.0),
    )

    finite = pivot[np.isfinite(pivot["Cl_Na_ratio"])]
    print(f"  Frames with finite ratio: {len(finite)} / {len(pivot)}")

    if len(finite) > 0:
        print(f"  Mean ratio : {finite['Cl_Na_ratio'].mean():.3f}")
        print(f"  Median     : {finite['Cl_Na_ratio'].median():.3f}")

    return pivot


# ═══════════════════════════════════════════════════════════════════════════
#  Raw data CSV export  (for repository deposition)
# ═══════════════════════════════════════════════════════════════════════════


def export_raw_data(
    combined_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    all_pairwise: Dict[str, Dict[Tuple[float, float], float]],
    ratio_df: pd.DataFrame,
) -> None:
    """Write all computed data to CSV files for deposition.

    Files produced
    --------------
    combined_folds_frame_data.csv
        The full pooled dataset (every frame × fold × pH × ion).
    combined_folds_summary_statistics.csv
        Grouped summary stats (pH × fold × ion).
    combined_folds_pairwise_tests.csv
        Mann–Whitney U p-values for each ion type.
    combined_folds_cl_na_ratios.csv
        Per-frame Cl⁻/Na⁺ ratios.
    """
    print("\nExporting raw data files …")

    # 1. Full combined dataset
    out1 = "combined_folds_frame_data.csv"
    combined_df.to_csv(out1, index=False)
    print(f"  {out1}")

    # 2. Summary statistics
    if not summary_df.empty:
        out2 = "combined_folds_summary_statistics.csv"
        summary_df.to_csv(out2, index=False)
        print(f"  {out2}")

    # 3. Pairwise test results
    pw_rows: list = []
    for ion, pairs in all_pairwise.items():
        for (ph1, ph2), p_val in pairs.items():
            pw_rows.append({
                "ion_type": ion,
                "pH_1": ph1,
                "pH_2": ph2,
                "mann_whitney_p": p_val,
            })
    if pw_rows:
        out3 = "combined_folds_pairwise_tests.csv"
        pd.DataFrame(pw_rows).to_csv(out3, index=False)
        print(f"  {out3}")

    # 4. Ratio table
    if not ratio_df.empty:
        out4 = "combined_folds_cl_na_ratios.csv"
        ratio_df.to_csv(out4, index=False)
        print(f"  {out4}")


# ═══════════════════════════════════════════════════════════════════════════
#  Main pipeline
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Load data, run statistics, generate plots, export CSVs."""
    print("Combined FOLD1 + FOLD2 Ion Interaction Analysis")
    print("=" * 60)

    # -- Load and combine all data ----------------------------------------
    combined_df = combine_all_data()
    if combined_df.empty:
        print("No data to analyse!")
        return

    print(f"Folds : {sorted(combined_df['fold'].unique())}")
    print(f"pH    : {sorted(combined_df['pH'].unique())}")
    print(f"Ions  : {sorted(combined_df['ion_type'].unique())}")

    # -- Summary statistics -----------------------------------------------
    summary_df = generate_summary_statistics(combined_df)

    # -- Per-ion analysis and plots ---------------------------------------
    all_pairwise: Dict[str, Dict[Tuple[float, float], float]] = {}

    for ion in ION_TYPES:
        print(f"\n{'=' * 60}")
        print(f"ANALYSING {ion} INTERACTIONS")
        print(f"{'=' * 60}")

        pw = perform_statistical_analysis(combined_df, ion)
        all_pairwise[ion] = pw
        create_violin_plot(combined_df, ion, pw)

    # -- Cl⁻ / Na⁺ ratio analysis ----------------------------------------
    ratio_df = pd.DataFrame()
    if set(ION_TYPES) <= set(combined_df["ion_type"].unique()):
        ratio_df = compute_cl_na_ratios(combined_df)

    # -- Export everything ------------------------------------------------
    export_raw_data(combined_df, summary_df, all_pairwise, ratio_df)

    print(f"\nAnalysis complete — all plots and statistics generated.")


if __name__ == "__main__":
    main()
