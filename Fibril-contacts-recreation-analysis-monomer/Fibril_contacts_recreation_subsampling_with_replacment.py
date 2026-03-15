#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fibril_contacts_recreation_subsampling_with_replacment.py
=====================================
Quantify how well monomer MD simulations recreate the inter-chain
contact patterns found in a fibril crystal structure (PDB 6WQK).

Workflow
--------
    1. Parse the fibril reference file
       (``6WQK_interactions_5p0A.txt``) and extract all unique
       **inter-chain** residue–residue contacts as a reference set.
    2. For each monomer time-series file
       (``*_TimeSeries_Protein_Direct_ALL_CONTACTS.txt``), count how
       many fibril reference contacts are recreated at each frame.
    3. Restrict to the last 600 ns (equilibrated region).
    4. Compute the autocorrelation function of the recreation count and
       determine the decorrelation time.
    5. Systematically sub-sample every *τ_decorr* frames to obtain
       statistically independent observations.
    6. Pool FOLD1 + FOLD2 data and produce violin plots with
       Mann–Whitney U significance brackets.

Outputs
-------
    * ``Fibril_Contacts_Violin_Plot.png``
    * ``Fibril_Contacts_Summary_Statistics.csv``
    * ``fibril_recreation_subsampled_data.csv``  *(new)*
      — the full sub-sampled dataset used for plotting, suitable for
      repository deposition.
    * ``fibril_recreation_pairwise_tests.csv``  *(new)*
      — Mann–Whitney U p-values and effect sizes.

Inputs expected in the working directory
----------------------------------------
    * ``6WQK_interactions_5p0A.txt``  — fibril reference contacts
    * ``FOLD<n>_pH<PH>_TimeSeries_Protein_Direct_ALL_CONTACTS.txt``

Usage
-----
    python Fibril_contacts_recreation_subsampling_with_replacment.py

Requirements
------------
    NumPy, pandas, matplotlib, scipy

Author  : Aleksandra Wosztyl (Rizo Lab, UT Southwestern Medical Center)
Created : 2026
"""

from __future__ import annotations

import datetime
import os
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu


# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════

# Reference fibril structure file
FIBRIL_REF_FILE: str = "6WQK_interactions_5p0A.txt"

# Only analyse the last N nanoseconds of each trajectory (equilibrated
# region).  Set to None to use all frames.
LAST_NS_ONLY: int = 600

# pH colour scheme for the violin bodies
PH_COLORS: Dict[float, str] = {
    4.0: "#bfff00",    # lime green  — acidic
    7.4: "#228B22",    # forest green — neutral
    8.5: "#008B8B",    # dark cyan   — alkaline
}


# ═══════════════════════════════════════════════════════════════════════════
#  Contact-line parsing helpers
# ═══════════════════════════════════════════════════════════════════════════

_RESIDUE_RE = re.compile(r"([A-Z]{3})(\d+)")


def _parse_contact_line(contact_line: str):
    """Extract (res1_name, res1_num, res2_name, res2_num) from a string
    like ``'ALA263-GLY305'``.  Returns four ``None``s on failure."""
    if "-" not in contact_line:
        return None, None, None, None

    try:
        res1, res2 = contact_line.split("-", 1)
        m1 = _RESIDUE_RE.match(res1.strip())
        m2 = _RESIDUE_RE.match(res2.strip())

        if m1 and m2:
            return (
                m1.group(1), int(m1.group(2)),
                m2.group(1), int(m2.group(2)),
            )
    except Exception:
        pass

    return None, None, None, None


def _is_valid_contact(
    res1_name: str, res1_num: int, res2_name: str, res2_num: int
) -> bool:
    """Return ``False`` for self-interactions (same name *and* number)."""
    return not (res1_name == res2_name and res1_num == res2_num)


def _make_contact_key(
    res1_name: str, res1_num: int, res2_name: str, res2_num: int
) -> str:
    """Canonical ``'RES1<num>-RES2<num>'`` key (sorted to avoid A-B / B-A
    duplicates)."""
    a = f"{res1_name}{res1_num}"
    b = f"{res2_name}{res2_num}"
    s = tuple(sorted((a, b)))
    return f"{s[0]}-{s[1]}"


# ═══════════════════════════════════════════════════════════════════════════
#  Fibril reference loading
# ═══════════════════════════════════════════════════════════════════════════


def load_fibril_reference_contacts(filename: str) -> Set[str]:
    """Parse the 6WQK interaction file and return the set of unique
    inter-chain contact keys."""
    print(f"Loading fibril reference contacts from {filename} …")

    if not os.path.exists(filename):
        print(f"  File not found!")
        return set()

    contacts: Set[str] = set()
    parsing = False

    with open(filename, "r") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("FORMAT:"):
                parsing = True
                continue

            if parsing and line and "[INTER-CHAIN]" in line:
                contact_part = line.split("[INTER-CHAIN]")[0].strip()
                r1n, r1i, r2n, r2i = _parse_contact_line(contact_part)

                if (
                    r1n is not None
                    and r2n is not None
                    and _is_valid_contact(r1n, r1i, r2n, r2i)
                ):
                    contacts.add(_make_contact_key(r1n, r1i, r2n, r2i))

    print(f"  {len(contacts)} unique fibril inter-chain contacts")
    return contacts


def parse_6wqk_contacts(filename: str) -> pd.DataFrame:
    """Load the fibril reference as a single-row DataFrame (for the
    reference-line annotation on the plot)."""
    print(f"Loading fibril reference structure from {filename} …")

    if not os.path.exists(filename):
        print(f"  File not found!")
        return pd.DataFrame()

    contacts: Set[str] = set()
    parsing = False

    with open(filename, "r") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("FORMAT:"):
                parsing = True
                continue

            if parsing and line and "[INTER-CHAIN]" in line:
                part = line.split("[INTER-CHAIN]")[0].strip()
                r1n, r1i, r2n, r2i = _parse_contact_line(part)
                if (
                    r1n is not None
                    and r2n is not None
                    and _is_valid_contact(r1n, r1i, r2n, r2i)
                ):
                    contacts.add(_make_contact_key(r1n, r1i, r2n, r2i))

    print(f"  {len(contacts)} unique fibril inter-chain contacts")

    if contacts:
        return pd.DataFrame({
            "contacts": [len(contacts)],
            "source": ["6WQK_fibril"],
            "pH": ["fibril"],
            "fold": ["reference"],
        })
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════
#  Autocorrelation & sub-sampling
# ═══════════════════════════════════════════════════════════════════════════


def _calculate_autocorrelation(
    series: np.ndarray, max_lag: int = 100
) -> np.ndarray:
    """Normalised autocorrelation function up to *max_lag*."""
    acf = [1.0]
    for lag in range(1, min(max_lag, len(series))):
        a, b = series[:-lag], series[lag:]
        if len(a) > 1 and np.var(a) > 0 and np.var(b) > 0:
            c = np.corrcoef(a, b)[0, 1]
            acf.append(c if np.isfinite(c) else 0.0)
        else:
            acf.append(0.0)
    return np.array(acf)


def _find_decorrelation_time(
    acf: np.ndarray, threshold: float = 0.1
) -> int:
    """First lag at which |ACF| drops below *threshold* (minimum 1)."""
    for i, c in enumerate(acf):
        if abs(c) < threshold:
            return max(1, i)
    return max(1, len(acf) - 1)


def _systematic_subsample(
    data: List, times: List, step: int
) -> Tuple[List, List]:
    """Keep every *step*-th element."""
    if step <= 1:
        return data, times
    return data[::step], times[::step]


# ═══════════════════════════════════════════════════════════════════════════
#  Time-series file parsing with sub-sampling
# ═══════════════════════════════════════════════════════════════════════════


def parse_time_series_with_autocorr(
    filename: str,
    reference_contacts: Set[str],
    last_ns_only: int | None = 600,
) -> List[int]:
    """Count fibril-matching contacts per frame and return a sub-sampled
    list of counts.

    Parameters
    ----------
    filename : str
        ``*_TimeSeries_Protein_Direct_ALL_CONTACTS.txt``
    reference_contacts : set
        Fibril inter-chain contact keys to match against.
    last_ns_only : int or None
        Restrict to the final *N* nanoseconds of the trajectory.
    """
    print(f"Parsing {filename} …")

    if not os.path.exists(filename):
        print(f"  File not found!")
        return []

    all_counts: List[int] = []
    all_times: List[float] = []
    current_contacts: Set[str] = set()
    current_time: float | None = None

    with open(filename, "r") as fh:
        for line in fh:
            line = line.strip()

            # Time header
            if line.startswith("t=") and "ns" in line:
                # Flush previous frame
                if current_time is not None:
                    n_match = len(
                        current_contacts.intersection(reference_contacts)
                    )
                    all_counts.append(n_match)
                    all_times.append(current_time)

                m = re.search(r"t=\s*(\d+(?:\.\d+)?)\s*ns", line)
                current_time = float(m.group(1)) if m else None
                current_contacts = set()

            elif (
                line
                and not line.startswith("t=")
                and not line.startswith("(no contacts")
                and "-" in line
            ):
                r1n, r1i, r2n, r2i = _parse_contact_line(line)
                if (
                    r1n is not None
                    and r2n is not None
                    and _is_valid_contact(r1n, r1i, r2n, r2i)
                ):
                    current_contacts.add(
                        _make_contact_key(r1n, r1i, r2n, r2i)
                    )

    # Flush last frame
    if current_time is not None:
        all_counts.append(
            len(current_contacts.intersection(reference_contacts))
        )
        all_times.append(current_time)

    # -- Restrict to last N ns ------------------------------------------------
    if last_ns_only and all_times:
        cutoff = max(all_times) - last_ns_only
        filtered = [
            (c, t)
            for c, t in zip(all_counts, all_times)
            if t >= cutoff
        ]
        if filtered:
            all_counts, all_times = list(zip(*filtered))
            all_counts, all_times = list(all_counts), list(all_times)

        print(f"  Filtered to last {last_ns_only} ns: "
              f"{len(all_counts)} frames")
        if all_times:
            print(f"  Time range: {min(all_times):.1f}–"
                  f"{max(all_times):.1f} ns")

    if len(all_counts) < 10:
        print(f"  Warning: only {len(all_counts)} frames")
        return all_counts

    # -- Autocorrelation analysis ---------------------------------------------
    print("  Computing autocorrelation …")
    acf = _calculate_autocorrelation(
        np.array(all_counts),
        max_lag=min(100, len(all_counts) // 2),
    )
    tau = _find_decorrelation_time(acf, threshold=0.1)

    print(f"  Decorrelation time: {tau} frames")
    print(f"  ACF(1) = {acf[1]:.3f}, "
          f"ACF(τ) = {acf[min(tau, len(acf) - 1)]:.3f}")

    # -- Systematic sub-sampling ----------------------------------------------
    sub_counts, sub_times = _systematic_subsample(
        all_counts, all_times, tau
    )

    print(f"  Sub-sampled: {len(sub_counts)} from "
          f"{len(all_counts)} frames "
          f"({len(all_counts) / len(sub_counts):.1f}×)")

    if sub_counts:
        total_ref = len(reference_contacts)
        mx = max(sub_counts)
        print(f"  Range: {min(sub_counts)}–{mx}, "
              f"mean: {np.mean(sub_counts):.1f}")
        print(f"  Recreation: {mx}/{total_ref} "
              f"({mx / total_ref * 100:.1f}%)")

    return sub_counts


def _extract_file_info(filename: str):
    """Return ``(fold, pH_float)`` from a filename like
    ``FOLD1_pH40_…``."""
    m = re.search(r"FOLD(\d+)_pH(\d+)_", filename)
    if m:
        fold = int(m.group(1))
        ph_code = int(m.group(2))
        ph_map = {40: 4.0, 74: 7.4, 85: 8.5}
        return fold, ph_map.get(ph_code, ph_code / 10.0)
    return None, None


# ═══════════════════════════════════════════════════════════════════════════
#  Data collection
# ═══════════════════════════════════════════════════════════════════════════


def collect_all_time_series_data(
    last_ns_only: int = 600,
) -> pd.DataFrame:
    """Discover time-series files, parse them with ACF sub-sampling,
    and return a pooled DataFrame."""
    print("\nLoading fibril reference contacts …")
    ref = load_fibril_reference_contacts(FIBRIL_REF_FILE)

    if not ref:
        print("No reference contacts — cannot proceed.")
        return pd.DataFrame()

    ts_files = sorted(
        str(p) for p in Path(".").glob(
            "*TimeSeries_Protein_Direct_ALL_CONTACTS.txt"
        )
    )
    print(f"Found {len(ts_files)} time-series files:")
    for f in ts_files:
        print(f"  {f}")

    rows: list = []
    for fname in ts_files:
        fold, ph = _extract_file_info(fname)
        if fold is None:
            print(f"  Could not parse info from {fname}")
            continue

        counts = parse_time_series_with_autocorr(
            fname, ref, last_ns_only
        )
        for c in counts:
            rows.append({
                "contacts": c,
                "pH": ph,
                "fold": fold,
                "source": f"FOLD{fold}_pH{ph}",
            })

    df = pd.DataFrame(rows)

    if not df.empty:
        print(f"\nSummary: {len(ref)} reference contacts, "
              f"{len(df)} data points, last {last_ns_only} ns")
        for ph in sorted(df["pH"].unique()):
            s = df.loc[df["pH"] == ph, "contacts"]
            print(f"  pH {ph}: {s.mean():.1f}±{s.std():.1f}, "
                  f"max {s.max()}, N={len(s)}")

    return df


# ═══════════════════════════════════════════════════════════════════════════
#  Statistical analysis
# ═══════════════════════════════════════════════════════════════════════════


def perform_statistical_analysis(
    df: pd.DataFrame,
) -> Dict[Tuple[float, float], float]:
    """Kruskal–Wallis + pairwise Mann–Whitney U.

    Returns ``{(pH_1, pH_2): p_value}`` and prints to console.
    """
    print(f"\nStatistical Analysis:")
    print("=" * 50)

    sim = df[df["pH"] != "fibril"].copy()
    ph_vals = sorted(sim["pH"].unique())

    if len(ph_vals) < 2:
        print("  Need ≥ 2 pH conditions")
        return {}

    groups = [
        sim.loc[sim["pH"] == ph, "contacts"].values for ph in ph_vals
    ]
    h, hp = kruskal(*groups)
    print(f"  Kruskal–Wallis: H = {h:.3f}, p = {hp:.4f}")

    print("\nDescriptive Statistics:")
    print("-" * 30)
    for ph in ph_vals:
        s = sim.loc[sim["pH"] == ph, "contacts"]
        print(f"  pH {ph}: mean={s.mean():.1f}, "
              f"median={s.median():.1f}, SD={s.std():.1f}, N={len(s)}")

    pairwise: Dict[Tuple[float, float], float] = {}
    print("\nPairwise Mann–Whitney U:")
    print("-" * 40)

    for i, ph1 in enumerate(ph_vals):
        for ph2 in ph_vals[i + 1:]:
            g1 = sim.loc[sim["pH"] == ph1, "contacts"]
            g2 = sim.loc[sim["pH"] == ph2, "contacts"]
            if len(g1) == 0 or len(g2) == 0:
                continue

            u, p = mannwhitneyu(g1, g2, alternative="two-sided")
            pairwise[(ph1, ph2)] = p

            n1, n2 = len(g1), len(g2)
            r = abs(1 - (2 * u) / (n1 * n2))
            mag = (
                "negligible" if r < 0.1
                else "small" if r < 0.3
                else "medium" if r < 0.5
                else "large"
            )
            print(f"  pH {ph1} vs {ph2}: "
                  f"p = {p:.4f}, r = {r:.3f} ({mag})")

    return pairwise


# ═══════════════════════════════════════════════════════════════════════════
#  Visualisation
# ═══════════════════════════════════════════════════════════════════════════


def _add_stat_annotation(
    ax, x1: int, x2: int, y: float, p_value: float,
    height_offset: float = 2,
) -> None:
    """Draw a significance bracket."""
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
        lw=1.2, c="black",
    )
    ax.text(
        (x1 + x2) * 0.5, y + height_offset, sym,
        ha="center", va="bottom", fontsize=12, fontweight="bold",
    )


def create_violin_plot(
    df: pd.DataFrame,
    fibril_df: pd.DataFrame | None = None,
) -> None:
    """Violin plot with jittered points coloured by fold."""
    plot_data = df[df["pH"] != "fibril"].copy()
    if plot_data.empty:
        print("No simulation data to plot!")
        return

    try:
        fig, ax = plt.subplots(figsize=(3, 5))
        ph_vals = sorted(plot_data["pH"].unique())

        # Violin bodies
        parts = ax.violinplot(
            [plot_data.loc[plot_data["pH"] == ph, "contacts"].values
             for ph in ph_vals],
            positions=range(len(ph_vals)),
            widths=0.7, showmeans=True, showmedians=True,
        )

        for pc, ph in zip(parts["bodies"], ph_vals):
            pc.set_facecolor(PH_COLORS.get(ph, "gray"))
            pc.set_alpha(0.7)
            pc.set_edgecolor("black")
            pc.set_linewidth(1)

        parts["cmeans"].set_color("red")
        parts["cmeans"].set_linewidth(2)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(2)
        for key in ("cbars", "cmins", "cmaxes"):
            parts[key].set_color("black")

        # Jittered strip points, coloured by fold
        fold_colors = {1: "darkorange", 2: "royalblue"}
        fold_markers = {1: "o", 2: "s"}

        for i, ph in enumerate(ph_vals):
            ph_slice = plot_data[plot_data["pH"] == ph]
            x_jitter = np.random.normal(i, 0.1, len(ph_slice))

            for fold in (1, 2):
                mask = ph_slice["fold"].values == fold
                if np.any(mask):
                    ax.scatter(
                        x_jitter[mask],
                        ph_slice["contacts"].values[mask],
                        c=fold_colors[fold],
                        marker=fold_markers[fold],
                        s=20, alpha=0.6,
                        edgecolors="black", linewidth=0.5,
                        label=f"FOLD{fold}" if i == 0 else "",
                    )

        # Significance annotations
        pw = perform_statistical_analysis(plot_data)
        if len(ph_vals) >= 2:
            y_max = plot_data["contacts"].max()
            configs = []

            if 4.0 in ph_vals and 7.4 in ph_vals:
                configs.append((
                    ph_vals.index(4.0), ph_vals.index(7.4),
                    y_max * 1.05, (4.0, 7.4),
                ))
            if 4.0 in ph_vals and 8.5 in ph_vals:
                configs.append((
                    ph_vals.index(4.0), ph_vals.index(8.5),
                    y_max * 1.09, (4.0, 8.5),
                ))
            if 7.4 in ph_vals and 8.5 in ph_vals:
                configs.append((
                    ph_vals.index(7.4), ph_vals.index(8.5),
                    y_max * 1.07, (7.4, 8.5),
                ))

            for x1, x2, h, pair in configs:
                if pair in pw:
                    _add_stat_annotation(
                        ax, x1, x2, h, pw[pair], y_max * 0.01
                    )

        ax.set_ylabel(
            "Fibril Contacts Recreated in Monomer (counts)",
            fontsize=12, fontweight="bold",
        )
        ax.set_xticks(range(len(ph_vals)))
        ax.set_xticklabels([f"pH {ph}" for ph in ph_vals])
        ax.legend(loc="lower right", frameon=True, fancybox=True)

        y_lo = max(0, plot_data["contacts"].min() - 1)
        y_hi = plot_data["contacts"].max() * 1.12
        ax.set_ylim(y_lo, y_hi)

        plt.tight_layout()

        outfile = "Fibril_Contacts_Violin_Plot.png"
        try:
            plt.savefig(outfile, dpi=300, bbox_inches="tight")
            print(f"  Plot → {outfile}")
        except PermissionError:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            alt = f"Fibril_Contacts_Violin_Plot_{ts}.png"
            plt.savefig(alt, dpi=300, bbox_inches="tight")
            print(f"  File locked; saved → {alt}")

        plt.show()

    except Exception as exc:
        print(f"Error creating plot: {exc}")


# ═══════════════════════════════════════════════════════════════════════════
#  Summary statistics
# ═══════════════════════════════════════════════════════════════════════════


def generate_summary_statistics(df: pd.DataFrame) -> None:
    """Print and save per-pH × per-fold summary statistics."""
    sim = df[df["pH"] != "fibril"].copy()
    if sim.empty:
        return

    rows: list = []
    for ph in sorted(sim["pH"].unique()):
        ph_slice = sim[sim["pH"] == ph]
        for fold in sorted(ph_slice["fold"].unique()):
            s = ph_slice.loc[ph_slice["fold"] == fold, "contacts"]
            rows.append({
                "pH": ph,
                "Fold": fold,
                "N_frames": len(s),
                "Mean": s.mean(),
                "Median": s.median(),
                "Std": s.std(),
                "Min": s.min(),
                "Max": s.max(),
                "Q25": s.quantile(0.25),
                "Q75": s.quantile(0.75),
            })

    summary = pd.DataFrame(rows)

    outfile = "Fibril_Contacts_Summary_Statistics.csv"
    try:
        summary.to_csv(outfile, index=False, float_format="%.2f")
        print(f"  Summary → {outfile}")
    except PermissionError:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        alt = f"Fibril_Contacts_Summary_Statistics_{ts}.csv"
        summary.to_csv(alt, index=False, float_format="%.2f")
        print(f"  File locked; saved → {alt}")

    print("\nSummary Statistics:")
    print(summary.to_string(index=False, float_format="{:.2f}".format))


# ═══════════════════════════════════════════════════════════════════════════
#  Raw data CSV export  (for repository deposition)
# ═══════════════════════════════════════════════════════════════════════════


def export_raw_data(
    df: pd.DataFrame,
    pairwise: Dict[Tuple[float, float], float],
) -> None:
    """Write the sub-sampled dataset and pairwise tests to CSV."""
    print("\nExporting raw data …")

    # 1. Full sub-sampled dataset
    sim = df[df["pH"] != "fibril"].copy()
    out1 = "fibril_recreation_subsampled_data.csv"
    sim.to_csv(out1, index=False)
    print(f"  {out1}  ({len(sim)} rows)")

    # 2. Pairwise test results
    if pairwise:
        pw_rows = []
        for (ph1, ph2), p in pairwise.items():
            g1 = sim.loc[sim["pH"] == ph1, "contacts"]
            g2 = sim.loc[sim["pH"] == ph2, "contacts"]
            u, _ = mannwhitneyu(g1, g2, alternative="two-sided")
            n1, n2 = len(g1), len(g2)
            r = abs(1 - (2 * u) / (n1 * n2))
            pw_rows.append({
                "pH_1": ph1,
                "pH_2": ph2,
                "mann_whitney_U": u,
                "p_value": p,
                "effect_size_r": round(r, 4),
            })
        out2 = "fibril_recreation_pairwise_tests.csv"
        pd.DataFrame(pw_rows).to_csv(out2, index=False)
        print(f"  {out2}")


# ═══════════════════════════════════════════════════════════════════════════
#  Main pipeline
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Execute the full fibril contact recreation analysis."""
    print("Fibril Contact Recreation Analysis — Monomer Simulations")
    print("=" * 60)
    print("  • Reference: 6WQK inter-chain contacts")
    print(f"  • Analysis window: last {LAST_NS_ONLY} ns (equilibrated)")
    print("  • ACF-based systematic sub-sampling")
    print("  • Unique pairs only (A-B = B-A), self-contacts excluded")
    print("  • FOLD1 + FOLD2 combined")

    # -- Collect sub-sampled data ---------------------------------------------
    df = collect_all_time_series_data(last_ns_only=LAST_NS_ONLY)

    if df.empty:
        print("No data found!")
        print("Expected: *TimeSeries_Protein_Direct_ALL_CONTACTS.txt")
        print(f"Expected: {FIBRIL_REF_FILE}")
        return

    print(f"Collected {len(df)} data points from "
          f"{df['source'].nunique()} sources")

    # -- Fibril reference for plot annotation ----------------------------------
    fibril_df = parse_6wqk_contacts(FIBRIL_REF_FILE)
    if not fibril_df.empty:
        print(f"Fibril reference: "
              f"{fibril_df['contacts'].iloc[0]} inter-chain contacts")

    # -- Summary statistics ---------------------------------------------------
    print("\nGenerating summary statistics …")
    generate_summary_statistics(df)

    # -- Violin plot ----------------------------------------------------------
    print("\nCreating violin plot …")
    create_violin_plot(df, fibril_df)

    # -- Raw data export ------------------------------------------------------
    pw = perform_statistical_analysis(df)
    export_raw_data(df, pw)

    print("\nDone.")
    print("  Fibril_Contacts_Violin_Plot.png")
    print("  Fibril_Contacts_Summary_Statistics.csv")
    print("  fibril_recreation_subsampled_data.csv")
    print("  fibril_recreation_pairwise_tests.csv")


if __name__ == "__main__":
    main()
