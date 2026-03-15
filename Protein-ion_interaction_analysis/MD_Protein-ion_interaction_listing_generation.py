#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MD_Protein-ion_interaction_listing_generation.py.py
=============================
Analyse protein–ion interactions on a per-frame basis from pre-computed
interaction files produced by ``extract_protein_ion_interactions.py``.

For each frame in the trajectory, the script counts the number of
**unique residues** that interact with at least one ion of a given type
(Cl⁻ or Na⁺).  Statistics are reported for the full trajectory and for
the last 100 frames, enabling comparison of equilibrated vs overall
behaviour.

Outputs per pH condition
------------------------
    * ``*_Frame_Analysis_Cl-.txt`` / ``*_Frame_Analysis_Na+.txt``
      — tab-delimited frame × ion_counts (all 750 frames, zeros included).
    * ``*_Frame_Analysis_Report.txt``
      — human-readable statistics report (total + last-100 comparison).
    * ``*_interactions_per_frame.png``
      — two-panel time series plot (Cl⁻ and Na⁺).
    * ``*_frame_interactions.csv``  *(new)*
      — machine-readable CSV combining both ion types for repository
      deposition.

Usage
-----
    python MD_Protein-ion_interaction_listing_generation.py.py

Adjust ``FOLD``, ``PH_CONDITIONS``, and ``CUTOFF`` below.

Requirements
------------
    NumPy, pandas, matplotlib

Author  : Aleksandra Wosztyl (Rizo Lab, UT Southwestern Medical Center)
Created : 2026
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════

CUTOFF: float = 5.0    # distance threshold (Å) used in input filenames
FOLD: str = "1-2"        # single-chain (monomer) system

# Total number of frames in each trajectory (used for zero-padding)
TOTAL_FRAMES: int = 750

PH_CONDITIONS: List[Dict[str, str]] = [
    {"PH": "40", "PHdot": "4.0"},
    {"PH": "74", "PHdot": "7.4"},
    {"PH": "85", "PHdot": "8.5"},
]


# ═══════════════════════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════════════════════


def load_ion_interactions(filename: str) -> pd.DataFrame:
    """Parse an interaction file into a DataFrame.

    The expected line format is::

        frame <n> <resname> <resid> chain SYSTEM interacts_with_<ion>_<ionresid>

    Returns
    -------
    pd.DataFrame
        Columns: ``frame``, ``resname``, ``resid``, ``ion_type``,
        ``ion_resid``, ``ion_id``, ``residue_id``.
    """
    if not os.path.exists(filename):
        print(f"ERROR: File '{filename}' not found.")
        return pd.DataFrame()

    print(f"Reading: {filename}")

    with open(filename, "r") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]

    pattern = re.compile(
        r"frame\s+(\d+)\s+(\w+)\s+(\d+)\s+.*?"
        r"interacts_with_(Cl-|Na\+)_(\d+)"
    )

    data: list = []
    for line in lines:
        m = pattern.search(line)
        if not m:
            continue
        frame, resname, resid, ion_type, ion_resid = m.groups()
        data.append({
            "frame": int(frame),
            "resname": resname,
            "resid": int(resid),
            "ion_type": ion_type,
            "ion_resid": int(ion_resid),
            "ion_id": f"{ion_type}_{ion_resid}",
            "residue_id": f"{resname} {resid}",
        })

    print(f"  Loaded {len(data)} interactions")
    return pd.DataFrame(data)


# ═══════════════════════════════════════════════════════════════════════════
#  Per-frame analysis
# ═══════════════════════════════════════════════════════════════════════════


def analyze_interactions_per_frame(df: pd.DataFrame) -> Dict[int, int]:
    """Count unique residues interacting with any ion per frame.

    Returns
    -------
    dict
        ``{frame_number: n_unique_residues}``.  Frames with zero
        interactions are *not* included (callers pad with zeros as
        needed).
    """
    if df.empty:
        return {}

    frame_interactions: Dict[int, int] = {}
    for frame, grp in df.groupby("frame"):
        frame_interactions[int(frame)] = grp["residue_id"].nunique()

    return frame_interactions


def get_frame_statistics(
    frame_interactions: Dict[int, int],
    total_frames: int | None = None,
) -> Dict[str, float]:
    """Compute summary statistics over frame interaction counts.

    Parameters
    ----------
    frame_interactions : dict
        ``{frame: count}`` — may be sparse (missing frames = 0).
    total_frames : int or None
        If given, zero-pad to this many frames before computing stats.
    """
    if not frame_interactions:
        return {
            "total_frames_analyzed": 0,
            "frames_with_interactions": 0,
            "total_interactions": 0,
            "avg_interactions_per_frame": 0.0,
            "avg_interactions_per_active_frame": 0.0,
            "max_interactions_per_frame": 0,
            "min_interactions_per_frame": 0,
        }

    # Zero-pad if a total frame count is provided
    if total_frames:
        counts = [
            frame_interactions.get(f, 0)
            for f in range(1, total_frames + 1)
        ]
    else:
        counts = list(frame_interactions.values())

    active = [c for c in counts if c > 0]

    return {
        "total_frames_analyzed": len(counts),
        "frames_with_interactions": len(active),
        "total_interactions": sum(counts),
        "avg_interactions_per_frame": float(np.mean(counts)),
        "avg_interactions_per_active_frame": (
            float(np.mean(active)) if active else 0.0
        ),
        "max_interactions_per_frame": max(counts),
        "min_interactions_per_frame": min(counts),
    }


def analyze_last_n_frames(
    frame_interactions: Dict[int, int],
    n_frames: int = 100,
) -> Tuple[Dict[int, int], List[int]]:
    """Extract the last *n_frames* from the interaction dict.

    Returns the filtered dict and the list of frame numbers used.
    """
    if not frame_interactions:
        return {}, []

    all_frames = sorted(frame_interactions.keys())

    if len(all_frames) < n_frames:
        print(f"  Warning: only {len(all_frames)} frames available, "
              f"using all instead of last {n_frames}")
        last = all_frames
    else:
        last = all_frames[-n_frames:]

    return {f: frame_interactions[f] for f in last}, last


# ═══════════════════════════════════════════════════════════════════════════
#  Legacy text-file output
# ═══════════════════════════════════════════════════════════════════════════


def generate_simple_data_files(
    frame_cl: Dict[int, int],
    frame_na: Dict[int, int],
    ph: str,
) -> List[str]:
    """Write tab-delimited frame × ion_counts files (zeros included).

    These are the same format produced by the original script.
    """
    files: List[str] = []

    for ion_label, interactions in [("Cl-", frame_cl), ("Na+", frame_na)]:
        fname = f"FOLD{FOLD}_pH{ph}_Frame_Analysis_{ion_label}.txt"
        with open(fname, "w") as f:
            f.write("frame\tion_counts\n")
            for frame in range(1, TOTAL_FRAMES + 1):
                f.write(f"{frame}\t{interactions.get(frame, 0)}\n")
        files.append(fname)
        print(f"  {ion_label} data → {fname}")

    return files


# ═══════════════════════════════════════════════════════════════════════════
#  Raw data CSV export  (for repository deposition)
# ═══════════════════════════════════════════════════════════════════════════


def export_frame_interactions_csv(
    frame_cl: Dict[int, int],
    frame_na: Dict[int, int],
    ph: str,
) -> Path:
    """Write a single CSV with both ion types for deposition.

    Columns: ``frame, cl_unique_residues, na_unique_residues``.
    All frames 1 … TOTAL_FRAMES are included (zeros for inactive frames).
    """
    outpath = Path(
        f"FOLD{FOLD}_pH{ph}_frame_interactions_{CUTOFF:.1f}A.csv"
    )

    rows = []
    for frame in range(1, TOTAL_FRAMES + 1):
        rows.append({
            "frame": frame,
            "cl_unique_residues": frame_cl.get(frame, 0),
            "na_unique_residues": frame_na.get(frame, 0),
        })

    df = pd.DataFrame(rows)
    df.to_csv(outpath, index=False)
    print(f"  CSV export → {outpath}")
    return outpath


# ═══════════════════════════════════════════════════════════════════════════
#  Statistics report
# ═══════════════════════════════════════════════════════════════════════════


def generate_frame_analysis_report(
    stats_cl: Dict,
    stats_na: Dict,
    last100_cl: Dict,
    last100_na: Dict,
    ph_dot: str,
    ph: str,
) -> str:
    """Write a plain-text statistics report comparing total vs last-100."""
    filename = f"FOLD{FOLD}_pH{ph}_Frame_Analysis_Report.txt"

    with open(filename, "w") as f:
        f.write(f"Ion Interactions Per Frame Analysis — pH {ph_dot}\n")
        f.write("=" * 60 + "\n")
        f.write("Unique residue–ion interactions counted per frame.\n")
        f.write("Each residue counted at most once per frame per ion type.\n")
        f.write("-" * 60 + "\n\n")

        def _write_block(label: str, cl: Dict, na: Dict) -> None:
            f.write(f"{label}\n")
            f.write("-" * 30 + "\n")
            for ion_name, s in [("Cl⁻", cl), ("Na⁺", na)]:
                f.write(f"{ion_name} Interactions:\n")
                f.write(f"  Frames analysed      : {s['total_frames_analyzed']}\n")
                f.write(f"  Frames with contacts : {s['frames_with_interactions']}\n")
                f.write(f"  Total interactions   : {s['total_interactions']}\n")
                f.write(f"  Avg / frame          : {s['avg_interactions_per_frame']:.2f}\n")
                f.write(f"  Avg / active frame   : {s['avg_interactions_per_active_frame']:.2f}\n")
                f.write(f"  Max / frame          : {s['max_interactions_per_frame']}\n")
                f.write(f"  Min / frame          : {s['min_interactions_per_frame']}\n\n")

        _write_block("TOTAL FRAMES ANALYSIS:", stats_cl, stats_na)
        _write_block("LAST 100 FRAMES ANALYSIS:", last100_cl, last100_na)

        # Comparison section
        f.write("COMPARISON (Total vs Last 100 frames):\n")
        f.write("-" * 40 + "\n")
        for ion_name, full, last in [
            ("Cl⁻", stats_cl, last100_cl),
            ("Na⁺", stats_na, last100_na),
        ]:
            f.write(f"{ion_name} Interactions:\n")
            if full["avg_interactions_per_frame"] > 0:
                change = (
                    (last["avg_interactions_per_frame"]
                     - full["avg_interactions_per_frame"])
                    / full["avg_interactions_per_frame"] * 100
                )
                f.write(f"  Change in average: {change:+.1f}%\n")
            f.write("\n")

    print(f"  Report → {filename}")
    return filename


# ═══════════════════════════════════════════════════════════════════════════
#  Visualisation
# ═══════════════════════════════════════════════════════════════════════════


def create_frame_analysis_plot(
    frame_cl: Dict[int, int],
    frame_na: Dict[int, int],
    ph_dot: str,
    output_dir: str = "plots",
) -> None:
    """Two-panel time series of per-frame ion interactions."""
    os.makedirs(output_dir, exist_ok=True)

    # Merge frame ranges from both ion types
    all_frames = sorted(
        set(frame_cl.keys()) | set(frame_na.keys())
    )
    if not all_frames:
        print("  No frames to plot")
        return

    cl_counts = [frame_cl.get(f, 0) for f in all_frames]
    na_counts = [frame_na.get(f, 0) for f in all_frames]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(
        all_frames, cl_counts, "b-",
        linewidth=1, alpha=0.7, label="Cl⁻ interactions",
    )
    ax1.set_ylabel("Cl⁻ Interactions per Frame")
    ax1.set_title(f"Ion Interactions per Frame — pH {ph_dot}")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(
        all_frames, na_counts, "r-",
        linewidth=1, alpha=0.7, label="Na⁺ interactions",
    )
    ax2.set_xlabel("Frame Number")
    ax2.set_ylabel("Na⁺ Interactions per Frame")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    ph_clean = ph_dot.replace(".", "")
    outfile = os.path.join(
        output_dir,
        f"FOLD{FOLD}_pH{ph_clean}_interactions_per_frame.png",
    )
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Plot → {outfile}")


# ═══════════════════════════════════════════════════════════════════════════
#  Main pipeline
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Iterate over pH conditions and produce all outputs."""
    print("Ion Interactions Per Frame Analysis")
    print(f"cutoff={CUTOFF} Å, fold={FOLD}, "
          f"total_frames={TOTAL_FRAMES}")
    print("Analysing total frames and last 100 frames\n")

    for condition in PH_CONDITIONS:
        ph = condition["PH"]
        ph_dot = condition["PHdot"]

        print(f"\n{'=' * 50}")
        print(f"  pH {ph_dot}")
        print(f"{'=' * 50}")

        # -- Input file paths -------------------------------------------------
        cla_file = (
            f"FOLD{FOLD}_pH{ph}_ion_residues_with_CLA_interactions_"
            f"{CUTOFF}A.txt"
        )
        sod_file = (
            f"FOLD{FOLD}_pH{ph}_ion_residues_with_SOD_interactions_"
            f"{CUTOFF}A.txt"
        )

        # -- Load interactions ------------------------------------------------
        cla_df = load_ion_interactions(cla_file)
        sod_df = load_ion_interactions(sod_file)

        # -- Per-frame unique-residue counts ----------------------------------
        frame_cl = analyze_interactions_per_frame(cla_df)
        frame_na = analyze_interactions_per_frame(sod_df)

        print(f"  Cl⁻: {len(frame_cl)} frames with interactions")
        print(f"  Na⁺: {len(frame_na)} frames with interactions")

        # -- Statistics: full trajectory --------------------------------------
        stats_cl = get_frame_statistics(frame_cl)
        stats_na = get_frame_statistics(frame_na)

        # -- Statistics: last 100 frames --------------------------------------
        last100_cl, _ = analyze_last_n_frames(frame_cl, 100)
        last100_na, _ = analyze_last_n_frames(frame_na, 100)
        last100_stats_cl = get_frame_statistics(last100_cl)
        last100_stats_na = get_frame_statistics(last100_na)

        # -- Legacy text data files -------------------------------------------
        data_files = generate_simple_data_files(frame_cl, frame_na, ph)

        # -- CSV export for repository deposition -----------------------------
        export_frame_interactions_csv(frame_cl, frame_na, ph)

        # -- Text report ------------------------------------------------------
        report = generate_frame_analysis_report(
            stats_cl, stats_na,
            last100_stats_cl, last100_stats_na,
            ph_dot, ph,
        )

        # -- Plot -------------------------------------------------------------
        try:
            create_frame_analysis_plot(frame_cl, frame_na, ph_dot)
        except Exception as exc:
            print(f"  Could not create plot: {exc}")

        # -- Console summary --------------------------------------------------
        print(f"\n  Summary (pH {ph_dot}):")
        print(f"    Cl⁻ total: {stats_cl['total_interactions']} "
              f"(avg {stats_cl['avg_interactions_per_frame']:.2f}/frame)")
        print(f"    Na⁺ total: {stats_na['total_interactions']} "
              f"(avg {stats_na['avg_interactions_per_frame']:.2f}/frame)")
        print(f"    Cl⁻ last 100: {last100_stats_cl['total_interactions']} "
              f"(avg {last100_stats_cl['avg_interactions_per_frame']:.2f}/frame)")
        print(f"    Na⁺ last 100: {last100_stats_na['total_interactions']} "
              f"(avg {last100_stats_na['avg_interactions_per_frame']:.2f}/frame)")
        print(f"    Data files: {', '.join(data_files)}")
        print(f"    Report: {report}")

    print("\nDone.")


if __name__ == "__main__":
    main()
