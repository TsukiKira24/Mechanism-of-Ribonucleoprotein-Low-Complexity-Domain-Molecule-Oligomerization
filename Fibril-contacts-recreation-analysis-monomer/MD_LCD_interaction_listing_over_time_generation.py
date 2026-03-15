#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MD_LCD_interaction_listing_over_time_generation.py
===============================
Frame-by-frame protein–protein contact analysis for monomer trajectories.

For every frame in a multi-model PDB trajectory, the script identifies
all residue–residue contacts (non-hydrogen atoms within a distance cutoff)
and writes the complete contact list per time point.  This captures the
**time evolution** of the contact network rather than a single aggregate
matrix.

Outputs per pH condition
------------------------
    * ``*_TimeSeries_Protein_Direct_ALL_CONTACTS.txt``
      — legacy text file with all contacts listed under each time header.
    * ``*_timeseries_contacts.csv``  *(new)*
      — machine-readable CSV for repository deposition: one row per
      contact per frame, with columns ``frame, time_ns, residue_1,
      residue_2``.
    * ``*_timeseries_per_frame_summary.csv``  *(new)*
      — per-frame contact counts.

Usage
-----
    python MD_LCD_interaction_listing_over_time_generation.py

Adjust ``FOLD``, ``PH_CONDITIONS``, and ``CUTOFF`` below.

Requirements
------------
    MDAnalysis, NumPy, pandas

Author  : Aleksandra Wosztyl (Rizo Lab, UT Southwestern Medical Center)
Created : 2026
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array


# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════

CUTOFF: float = 5.0       # distance threshold (Å)
FOLD: str = "1"            # monomer system

MATRIX_SIZE: int = 154     # number of residues in the chain
START_RESIDUE: int = 188   # first residue in PDB display numbering

PH_CONDITIONS: List[Dict[str, str]] = [
    {"PH": "40", "PHdot": "4.0"},
    {"PH": "74", "PHdot": "7.4"},
    {"PH": "85", "PHdot": "8.5"},
]


# ═══════════════════════════════════════════════════════════════════════════
#  Index / naming helpers
# ═══════════════════════════════════════════════════════════════════════════


def _index_to_display_resnum(idx: int) -> int:
    """Convert a 0-based matrix index to the PDB display residue number."""
    return START_RESIDUE + idx


def _get_residue_name_from_pdb(pdb_file: str, residue_num: int) -> str:
    """Read the 3-letter residue name for *residue_num* from a PDB file.

    Scans ATOM records until it finds a matching ``resid``; returns
    ``'UNK'`` if nothing is found.
    """
    try:
        with open(pdb_file, "r") as fh:
            for line in fh:
                if line.startswith("ATOM"):
                    try:
                        if int(line[22:26].strip()) == residue_num:
                            return line[17:20].strip()
                    except ValueError:
                        continue
    except Exception:
        pass
    return "UNK"


def _create_residue_mapping(pdb_file: str) -> Dict[int, str]:
    """Build a mapping ``{matrix_index: 3-letter residue name}``."""
    mapping: Dict[int, str] = {}
    for i in range(MATRIX_SIZE):
        data_resnum = i + 1  # 1-based numbering in the PDB
        mapping[i] = _get_residue_name_from_pdb(pdb_file, data_resnum)
    return mapping


def _display_name(idx: int, mapping: Dict[int, str]) -> str:
    """Return a label like ``'ALA205'`` for a matrix index."""
    name = mapping.get(idx, "UNK")
    return f"{name}{_index_to_display_resnum(idx)}"


# ═══════════════════════════════════════════════════════════════════════════
#  PDB timestamp extraction
# ═══════════════════════════════════════════════════════════════════════════


def _extract_pdb_timestamps(pdb_filename: str) -> List[float]:
    """Parse ``t= <value>`` from TITLE records in a multi-model PDB.

    Returns a list of time values in nanoseconds.  If fewer timestamps
    than frames are found, the caller falls back to frame-index numbering.
    """
    timestamps: List[float] = []

    if not os.path.exists(pdb_filename):
        print(f"ERROR: PDB file '{pdb_filename}' not found.")
        return timestamps

    pattern = re.compile(r"t=\s*(\d+\.?\d*)")

    with open(pdb_filename, "r") as fh:
        for line in fh:
            if line.startswith("TITLE"):
                m = pattern.search(line)
                if m:
                    time_ps = float(m.group(1))
                    timestamps.append(time_ps / 1000.0)  # ps → ns

    return timestamps


# ═══════════════════════════════════════════════════════════════════════════
#  Frame-by-frame contact extraction
# ═══════════════════════════════════════════════════════════════════════════


def process_pdb_contacts_per_frame(
    pdb_filename: str,
    residue_mapping: Dict[int, str],
) -> Dict[float, List[str]]:
    """Compute all residue–residue contacts for every frame.

    Returns
    -------
    dict
        ``{time_ns: [sorted list of "RES1<num>-RES2<num>" strings]}``.
        Frames with no contacts map to an empty list.
    """
    if not os.path.exists(pdb_filename):
        print(f"ERROR: PDB file '{pdb_filename}' not found.")
        return {}

    # Attempt to extract real timestamps from PDB TITLE records
    timestamps = _extract_pdb_timestamps(pdb_filename)

    u = mda.Universe(pdb_filename, format="PDB")
    n_frames = len(u.trajectory)

    print(f"  {n_frames} frames in PDB, "
          f"{len(timestamps)} timestamps from headers")

    # Fall back to 1-based integer numbering if timestamps are incomplete
    if len(timestamps) < n_frames:
        print("  Using frame index + 1 as nanosecond number")
        timestamps = [float(i + 1) for i in range(n_frames)]

    frame_contacts: Dict[float, List[str]] = {}

    for frame_idx, _ts in enumerate(u.trajectory):
        if frame_idx % 10 == 0:
            print(f"  Frame {frame_idx + 1}/{n_frames}")

        frame_time = (
            timestamps[frame_idx]
            if frame_idx < len(timestamps)
            else float(frame_idx + 1)
        )

        # Select non-hydrogen protein atoms (fallback to all non-H)
        try:
            atoms = u.select_atoms("protein and not name H*")
        except Exception:
            atoms = u.select_atoms("not name H*")

        if atoms.n_atoms == 0:
            frame_contacts[frame_time] = []
            continue

        resids = atoms.resids - 1  # 0-based matrix indices
        positions = atoms.positions

        # Keep only atoms whose resid falls inside the matrix
        valid = (resids >= 0) & (resids < MATRIX_SIZE)
        if not np.any(valid):
            frame_contacts[frame_time] = []
            continue

        resids = resids[valid]
        positions = positions[valid]

        # All-vs-all distances → boolean contact matrix
        dists = distance_array(positions, positions)
        in_contact = dists < CUTOFF

        # Collapse to unique residue-level pairs
        ci, cj = np.where(in_contact)
        contacted_pairs: set = set()

        for a, b in zip(ci, cj):
            if a == b:
                continue
            ri, rj = resids[a], resids[b]
            if ri != rj and 0 <= ri < MATRIX_SIZE and 0 <= rj < MATRIX_SIZE:
                contacted_pairs.add(tuple(sorted((ri, rj))))

        # Convert to human-readable labels
        contact_list = sorted(
            f"{_display_name(ri, residue_mapping)}-"
            f"{_display_name(rj, residue_mapping)}"
            for ri, rj in contacted_pairs
        )

        frame_contacts[frame_time] = contact_list

    return frame_contacts


# ═══════════════════════════════════════════════════════════════════════════
#  Legacy text output
# ═══════════════════════════════════════════════════════════════════════════


def write_legacy_contact_file(
    frame_contacts: Dict[float, List[str]],
    fold: str,
    ph: str,
) -> str:
    """Write the ``ALL_CONTACTS`` text file (original format).

    Each time point gets a header line followed by one contact per line.
    """
    filename = (
        f"FOLD{fold}_pH{ph}_TimeSeries_Protein_Direct_ALL_CONTACTS.txt"
    )

    with open(filename, "w") as f:
        for frame_time in sorted(frame_contacts):
            # Integer label when the time is a whole number
            if frame_time == int(frame_time):
                f.write(f"t= {int(frame_time)} ns\n")
            else:
                f.write(f"t= {frame_time:.1f} ns\n")

            contacts = frame_contacts[frame_time]
            if contacts:
                for c in contacts:
                    f.write(f"{c}\n")
            else:
                f.write("(no contacts found)\n")

            f.write("\n")

    print(f"  Legacy text → {filename}")
    return filename


# ═══════════════════════════════════════════════════════════════════════════
#  Raw data CSV export  (for repository deposition)
# ═══════════════════════════════════════════════════════════════════════════


def export_contacts_csv(
    frame_contacts: Dict[float, List[str]],
    fold: str,
    ph: str,
) -> Path:
    """Write one CSV row per contact per frame.

    Columns: ``frame_time_ns, residue_1, residue_2``.
    """
    outpath = Path(
        f"FOLD{fold}_pH{ph}_timeseries_contacts_{CUTOFF:.1f}A.csv"
    )

    rows: list = []
    for t in sorted(frame_contacts):
        for contact in frame_contacts[t]:
            # Contact string is "RES1<num>-RES2<num>"
            parts = contact.split("-")
            if len(parts) == 2:
                rows.append({
                    "frame_time_ns": t,
                    "residue_1": parts[0],
                    "residue_2": parts[1],
                })

    df = pd.DataFrame(rows)
    df.to_csv(outpath, index=False)
    print(f"  Pairwise CSV → {outpath}  ({len(df)} rows)")
    return outpath


def export_per_frame_summary_csv(
    frame_contacts: Dict[float, List[str]],
    fold: str,
    ph: str,
) -> Path:
    """Write per-frame contact counts for easy plotting."""
    outpath = Path(
        f"FOLD{fold}_pH{ph}_timeseries_per_frame_summary_{CUTOFF:.1f}A.csv"
    )

    rows = [
        {"frame_time_ns": t, "n_contacts": len(frame_contacts[t])}
        for t in sorted(frame_contacts)
    ]

    df = pd.DataFrame(rows)
    df.to_csv(outpath, index=False)
    print(f"  Per-frame CSV → {outpath}")
    return outpath


# ═══════════════════════════════════════════════════════════════════════════
#  Main pipeline
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Iterate over pH conditions and extract per-frame contacts."""
    print("Frame-by-Frame Protein Contact Analysis")
    print(f"cutoff={CUTOFF} Å, matrix_size={MATRIX_SIZE}, fold={FOLD}")
    print("Extracting ALL protein–protein contacts in EVERY frame\n")

    for condition in PH_CONDITIONS:
        ph = condition["PH"]
        ph_dot = condition["PHdot"]

        print(f"\n{'=' * 60}")
        print(f"  pH {ph_dot}")
        print(f"{'=' * 60}")

        pdb_file = f"FOLD{FOLD}_pH{ph}_ion_1ns.pdb"

        if not os.path.exists(pdb_file):
            print(f"  [SKIP] PDB not found: {pdb_file}")
            continue

        # Build residue name mapping from the first frame of the PDB
        residue_mapping = _create_residue_mapping(pdb_file)

        # Extract contacts for every frame
        print("  Analysing protein–protein contacts per frame …")
        frame_contacts = process_pdb_contacts_per_frame(
            pdb_file, residue_mapping
        )

        if not frame_contacts:
            print("  No frame contact data generated")
            continue

        # -- Legacy text file -------------------------------------------------
        write_legacy_contact_file(frame_contacts, FOLD, ph)

        # -- CSV exports for repository deposition ----------------------------
        export_contacts_csv(frame_contacts, FOLD, ph)
        export_per_frame_summary_csv(frame_contacts, FOLD, ph)

        # -- Console summary --------------------------------------------------
        n_frames = len(frame_contacts)
        total = sum(len(c) for c in frame_contacts.values())
        avg = total / n_frames if n_frames else 0

        times = sorted(frame_contacts)
        t_range = times[-1] - times[0] if times else 0

        print(f"\n  Frames analysed     : {n_frames}")
        print(f"  Total contacts      : {total}")
        print(f"  Avg per frame       : {avg:.1f}")
        print(f"  Time range          : {times[0]:.1f}–{times[-1]:.1f} ns "
              f"({t_range:.1f} ns)")

        # Frame with the most contacts
        best_t, best_c = max(
            frame_contacts.items(), key=lambda x: len(x[1])
        )
        print(f"  Most contacts       : t={best_t:.1f} ns "
              f"({len(best_c)} contacts)")

    print("\nDone.")


if __name__ == "__main__":
    main()
