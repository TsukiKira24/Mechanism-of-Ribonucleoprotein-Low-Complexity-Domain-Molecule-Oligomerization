#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contact_map_DIMERS_fragment-263-319_and_full_fragment.py
==================================
Contact analysis pipeline for **dimer** (two-chain) molecular dynamics
trajectories.

This is the two-chain counterpart of ``contact_analysis_pipeline.py``
(monomer).  The matrix is doubled in size (chain A + chain B concatenated
along both axes), so intra-chain and inter-chain contacts are visible in
a single heatmap.

Three classes of contacts are computed per pH condition:

    1. **Direct contacts** – non-hydrogen protein atoms within a distance
       cutoff, computed frame-by-frame from a multi-model PDB via
       MDAnalysis.
    2. **Cl⁻-mediated contacts** – residue pairs bridged by the same
       chloride ion (parsed from pre-computed interaction files).
    3. **Na⁺-mediated contacts** – analogous to (2) for sodium ions.

Both **full** and **zoomed** (residues 263–319) contact maps are produced
for each pH.

Outputs per pH condition
------------------------
    * 3-panel heatmaps (full + zoomed) as PNG files.
    * Console summary with top-contacted residues.
    * **Raw data CSV files** for repository deposition: pairwise contact
      lists, per-residue counts, and full matrices.

Residue numbering
-----------------
The interaction files use a 1–154 numbering scheme (``data_start_residue``
to ``data_end_residue``).  Display labels on plots are shifted to the PDB
numbering 188–341 (``display_start_residue`` to ``display_end_residue``).

Frame filtering
---------------
Set ``USE_LAST_N_FRAMES`` to an integer (e.g. 50) to restrict analysis to
the final *N* frames of each trajectory.  Set to ``None`` to use all frames.

Usage
-----
    python Contact_map_DIMERS_fragment-263-319_and_full_fragment.py

Requirements
------------
    MDAnalysis, NumPy, pandas, matplotlib, seaborn

Author  : Aleksandra Wosztyl
"""

from __future__ import annotations

import os
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap

import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array


# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════

CUTOFF: float = 5.0    # distance threshold (Å)
FOLD: str = "1-2"

# Residue numbering in the interaction files (1-based internal numbering)
DATA_START_RESIDUE: int = 1
DATA_END_RESIDUE: int = 154

# Residue numbering on plot axes (PDB numbering)
DISPLAY_START_RESIDUE: int = 188
DISPLAY_END_RESIDUE: int = 341

# Frame filtering — set to an integer to use only the last N frames,
# or None to process the entire trajectory.
USE_LAST_N_FRAMES: Optional[int] = None

# Derived constants
RESIDUES_PER_CHAIN: int = DATA_END_RESIDUE - DATA_START_RESIDUE + 1
MATRIX_SIZE: int = 2 * RESIDUES_PER_CHAIN  # chains A + B

# pH conditions to process
PH_CONDITIONS: List[Dict[str, str]] = [
    {"PH": "40", "PHdot": "4.0"},
    {"PH": "74", "PHdot": "7.4"},
    {"PH": "85", "PHdot": "8.5"},
]


# ═══════════════════════════════════════════════════════════════════════════
#  Index helpers
# ═══════════════════════════════════════════════════════════════════════════


def residue_index(residue_id: str) -> Optional[int]:
    """Convert a residue ID like ``'A_42'`` to a 0-based matrix index.

    Chain A residues occupy indices 0 … RESIDUES_PER_CHAIN-1,
    chain B residues occupy RESIDUES_PER_CHAIN … MATRIX_SIZE-1.
    Returns ``None`` if the residue is outside the expected range.
    """
    try:
        chain, resi_str = residue_id.split("_")
        resi = int(resi_str)
        if not (DATA_START_RESIDUE <= resi <= DATA_END_RESIDUE):
            return None
        offset = 0 if chain == "A" else RESIDUES_PER_CHAIN
        return offset + (resi - DATA_START_RESIDUE)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════════════════════


def load_ion_interactions(
    filename: str,
    expected_ion_type: str,
    last_n_frames: Optional[int] = None,
) -> pd.DataFrame:
    """Parse a whitespace-delimited interaction file into a DataFrame.

    The expected line format is::

        frame <n> <resname> <resid> chain <chain> interacts_with_<ion>_<ionresid>

    Parameters
    ----------
    filename : str
        Path to the interaction file.
    expected_ion_type : str
        ``"CLA"`` or ``"SOD"`` — used only for log messages.
    last_n_frames : int or None
        If set, keep only rows from the final *N* frames.

    Returns
    -------
    pd.DataFrame
        Columns: ``frame``, ``resname``, ``resid``, ``chain``,
        ``ion_type``, ``ion_resid``, ``ion_id``, ``residue_id``.
    """
    if not os.path.exists(filename):
        print(f"ERROR: File '{filename}' not found.")
        return pd.DataFrame()

    print(f"Reading {expected_ion_type} contact file: {filename}")

    with open(filename, "r") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]

    print(f"  Total lines: {len(lines)}")

    data: list = []
    pattern = re.compile(
        r"frame\s+(\d+)\s+(\w+)\s+(\d+)\s+chain\s+(\w)\s+"
        r"interacts_with_(Cl-|Na\+)_([0-9]+)"
    )

    for line in lines:
        m = pattern.match(line)
        if not m:
            continue
        frame, resname, resid, chain, ion_type, ion_resid = m.groups()
        data.append({
            "frame": int(frame),
            "resname": resname,
            "resid": int(resid),
            "chain": chain,
            "ion_type": ion_type,
            "ion_resid": int(ion_resid),
            "ion_id": f"{ion_type}_{ion_resid}",
            "residue_id": f"{chain}_{resid}",
        })

    df = pd.DataFrame(data)

    # Optionally restrict to the last N frames
    if last_n_frames is not None and len(df) > 0:
        max_frame = df["frame"].max()
        min_frame = max_frame - last_n_frames + 1
        df = df[df["frame"] >= min_frame]
        print(f"  Filtered to last {last_n_frames} frames "
              f"({min_frame}–{max_frame})")

    print(f"  Loaded {len(df)} {expected_ion_type} interactions")

    if len(df) > 0:
        res_nums = [int(rid.split("_")[1]) for rid in df["residue_id"]]
        print(f"  {df['ion_id'].nunique()} unique ions, "
              f"{df['residue_id'].nunique()} unique residues, "
              f"{df['frame'].nunique()} frames")
        print(f"  Residue range: {min(res_nums)}–{max(res_nums)}")

    return df


# ═══════════════════════════════════════════════════════════════════════════
#  Contact calculation
# ═══════════════════════════════════════════════════════════════════════════


def calculate_ion_contacts(
    df: pd.DataFrame,
) -> Dict[Tuple[str, str], int]:
    """Identify residue pairs bridged by the same ion in the same frame.

    Returns a dict ``{(residue_id_A, residue_id_B): count}``.
    """
    print("Calculating ion-mediated contacts …")

    contacts: Dict[Tuple[str, str], int] = defaultdict(int)

    for _frame, frame_grp in df.groupby("frame"):
        for _ion, ion_grp in frame_grp.groupby("ion_id"):
            residues = sorted(ion_grp["residue_id"].unique())
            if len(residues) < 2:
                continue
            for r1, r2 in combinations(residues, 2):
                contacts[tuple(sorted((r1, r2)))] += 1

    print(f"  {len(contacts)} unique pairs, "
          f"{sum(contacts.values())} total contacts")
    return contacts


def calculate_direct_ion_interactions(
    df: pd.DataFrame,
) -> Dict[str, int]:
    """Count how many atom-level ion contacts each residue has.

    This is the simpler per-residue metric (not pairwise).
    """
    print("Calculating direct residue–ion interactions …")

    counts: Dict[str, int] = defaultdict(int)
    for _, row in df.iterrows():
        counts[row["residue_id"]] += 1

    print(f"  {len(counts)} residues, {sum(counts.values())} interactions")
    return counts


def ion_contacts_to_array(contact_dict: Dict[str, int]) -> np.ndarray:
    """Map per-residue ion interaction counts into a length-MATRIX_SIZE array."""
    arr = np.zeros(MATRIX_SIZE, dtype=int)
    for residue_id, count in contact_dict.items():
        idx = residue_index(residue_id)
        if idx is not None:
            arr[idx] = count
    return arr


def contacts_to_matrix(
    contact_dict: Dict[Tuple[str, str], int],
) -> np.ndarray:
    """Convert a pairwise contact dict into a symmetric count matrix."""
    mat = np.zeros((MATRIX_SIZE, MATRIX_SIZE), dtype=int)
    for (r1, r2), count in contact_dict.items():
        i = residue_index(r1)
        j = residue_index(r2)
        if i is not None and j is not None:
            mat[i, j] = mat[j, i] = count
    return mat


# ═══════════════════════════════════════════════════════════════════════════
#  Direct contacts from PDB trajectory
# ═══════════════════════════════════════════════════════════════════════════


def process_pdb_single_pass(
    pdb_filename: str,
    last_n_frames: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute direct residue–residue contacts from a multi-model PDB.

    Chain assignment heuristic: if the atom has a ``chainid`` or ``segid``
    attribute, use it; otherwise assign chain A for resid < 200, chain B
    otherwise.

    Returns
    -------
    contact_counts : ndarray, shape (MATRIX_SIZE, MATRIX_SIZE)
        Symmetric matrix counting how many frames each pair is in contact.
    time_matrix : ndarray, shape (MATRIX_SIZE, n_frames_processed)
        Binary matrix flagging which residues participate in any contact
        at each frame.
    """
    if not os.path.exists(pdb_filename):
        print(f"ERROR: PDB file '{pdb_filename}' not found.")
        return (
            np.zeros((MATRIX_SIZE, MATRIX_SIZE), dtype=int),
            np.zeros((MATRIX_SIZE, 1), dtype=int),
        )

    print(f"Loading structure: {pdb_filename}")
    u = mda.Universe(pdb_filename, format="PDB")
    n_frames = len(u.trajectory)

    # Determine which frames to process
    if last_n_frames is not None and last_n_frames < n_frames:
        start_frame = n_frames - last_n_frames
        frames_to_process = range(start_frame, n_frames)
        print(f"  Processing last {last_n_frames} frames "
              f"({start_frame}–{n_frames - 1})")
    else:
        frames_to_process = range(n_frames)
        print(f"  {n_frames} frames — processing all")

    contact_counts = np.zeros((MATRIX_SIZE, MATRIX_SIZE), dtype=int)
    time_matrix = np.zeros((MATRIX_SIZE, len(frames_to_process)), dtype=int)

    for time_idx, frame_idx in enumerate(frames_to_process):
        if time_idx % 10 == 0:
            print(f"  Frame {frame_idx + 1}/{n_frames} "
                  f"(position {time_idx + 1}/{len(frames_to_process)})")

        u.trajectory[frame_idx]

        # Select non-hydrogen protein atoms (fallback to all non-H)
        try:
            protein_atoms = u.select_atoms("protein and not name H*")
        except Exception:
            protein_atoms = u.select_atoms("not name H*")

        if len(protein_atoms) == 0:
            print(f"  WARNING: No atoms in frame {frame_idx}")
            continue

        # Build per-atom residue IDs with chain assignment
        residue_ids: List[str] = []
        for atom in protein_atoms:
            if hasattr(atom, "chainid") and atom.chainid.strip():
                chain = atom.chainid.strip()
            elif hasattr(atom, "segid") and atom.segid.strip():
                chain = atom.segid.strip()
            else:
                chain = "A" if atom.resid < 200 else "B"
            residue_ids.append(f"{chain}_{atom.resid}")

        # All-vs-all distance matrix → boolean contact matrix
        positions = protein_atoms.positions
        dists = distance_array(positions, positions)
        in_contact = dists < CUTOFF

        frame_contacts = np.zeros((MATRIX_SIZE, MATRIX_SIZE), dtype=bool)
        active_residues: set = set()

        ci, cj = np.where(in_contact)
        for i, j in zip(ci, cj):
            if i == j:
                continue
            ri, rj = residue_ids[i], residue_ids[j]
            ii = residue_index(ri)
            jj = residue_index(rj)
            if ii is not None and jj is not None and ii != jj:
                frame_contacts[ii, jj] = True
                frame_contacts[jj, ii] = True
                active_residues.update((ii, jj))

        contact_counts += frame_contacts.astype(int)
        for res in active_residues:
            time_matrix[res, time_idx] = 1

    print("  PDB processing complete")
    return contact_counts, time_matrix


def calculate_residue_contact_counts(
    matrix: np.ndarray,
    ion_array: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Per-residue contact counts — row-sum of matrix, or ion array if given."""
    if ion_array is not None:
        return ion_array
    return np.sum(matrix, axis=1)


# ═══════════════════════════════════════════════════════════════════════════
#  Raw data export  (for repository deposition)
# ═══════════════════════════════════════════════════════════════════════════


def export_pairwise_contacts_csv(
    matrix: np.ndarray,
    label: str,
    fold: str,
    ph: str,
) -> Path:
    """Write non-zero upper-triangle entries of *matrix* to CSV.

    Each row contains ``chain_i, residue_i_data, residue_i_display,
    chain_j, residue_j_data, residue_j_display, contact_count``.
    """
    outpath = Path(
        f"FOLD{fold}_pH{ph}_{label}_pairwise_contacts_{CUTOFF:.1f}A.csv"
    )
    rows: list = []

    for i in range(MATRIX_SIZE):
        for j in range(i + 1, MATRIX_SIZE):
            if matrix[i, j] == 0:
                continue
            chain_i = "A" if i < RESIDUES_PER_CHAIN else "B"
            chain_j = "A" if j < RESIDUES_PER_CHAIN else "B"
            data_i = DATA_START_RESIDUE + (i % RESIDUES_PER_CHAIN)
            data_j = DATA_START_RESIDUE + (j % RESIDUES_PER_CHAIN)
            disp_i = DISPLAY_START_RESIDUE + (i % RESIDUES_PER_CHAIN)
            disp_j = DISPLAY_START_RESIDUE + (j % RESIDUES_PER_CHAIN)
            rows.append({
                "chain_i": chain_i,
                "residue_i_data": data_i,
                "residue_i_display": disp_i,
                "chain_j": chain_j,
                "residue_j_data": data_j,
                "residue_j_display": disp_j,
                "contact_count": int(matrix[i, j]),
            })

    df = pd.DataFrame(rows)
    df.to_csv(outpath, index=False)
    print(f"  Exported {len(df)} {label} pairwise contacts → {outpath}")
    return outpath


def export_per_residue_counts_csv(
    direct_counts: np.ndarray,
    cla_array: np.ndarray,
    sod_array: np.ndarray,
    fold: str,
    ph: str,
) -> Path:
    """Write per-residue contact counts (all three types) to CSV."""
    outpath = Path(
        f"FOLD{fold}_pH{ph}_per_residue_contact_counts_{CUTOFF:.1f}A.csv"
    )
    rows: list = []

    for idx in range(MATRIX_SIZE):
        chain = "A" if idx < RESIDUES_PER_CHAIN else "B"
        data_res = DATA_START_RESIDUE + (idx % RESIDUES_PER_CHAIN)
        disp_res = DISPLAY_START_RESIDUE + (idx % RESIDUES_PER_CHAIN)
        rows.append({
            "chain": chain,
            "residue_data": data_res,
            "residue_display": disp_res,
            "direct_contacts": int(direct_counts[idx]),
            "cla_ion_interactions": int(cla_array[idx]),
            "sod_ion_interactions": int(sod_array[idx]),
        })

    df = pd.DataFrame(rows)
    df.to_csv(outpath, index=False)
    print(f"  Exported per-residue counts → {outpath}")
    return outpath


def export_full_matrix_csv(
    matrix: np.ndarray,
    label: str,
    fold: str,
    ph: str,
) -> Path:
    """Write the complete N×N contact matrix with chain_residue headers."""
    outpath = Path(
        f"FOLD{fold}_pH{ph}_{label}_full_matrix_{CUTOFF:.1f}A.csv"
    )

    labels = []
    for idx in range(MATRIX_SIZE):
        chain = "A" if idx < RESIDUES_PER_CHAIN else "B"
        disp = DISPLAY_START_RESIDUE + (idx % RESIDUES_PER_CHAIN)
        labels.append(f"{chain}_{disp}")

    df = pd.DataFrame(matrix, index=labels, columns=labels)
    df.index.name = "residue"
    df.to_csv(outpath)
    print(f"  Exported full {label} matrix → {outpath}")
    return outpath


def export_time_matrix_csv(
    time_matrix: np.ndarray,
    fold: str,
    ph: str,
) -> Path:
    """Write the binary residue × frame direct-contact time matrix."""
    outpath = Path(
        f"FOLD{fold}_pH{ph}_direct_contact_time_matrix_{CUTOFF:.1f}A.csv"
    )

    row_labels = []
    for idx in range(MATRIX_SIZE):
        chain = "A" if idx < RESIDUES_PER_CHAIN else "B"
        disp = DISPLAY_START_RESIDUE + (idx % RESIDUES_PER_CHAIN)
        row_labels.append(f"{chain}_{disp}")

    col_labels = [f"frame_{i + 1}" for i in range(time_matrix.shape[1])]

    df = pd.DataFrame(time_matrix, index=row_labels, columns=col_labels)
    df.index.name = "residue"
    df.to_csv(outpath)
    print(f"  Exported time matrix → {outpath}")
    return outpath


# ═══════════════════════════════════════════════════════════════════════════
#  Visualisation helpers
# ═══════════════════════════════════════════════════════════════════════════


def create_custom_colormap(base_cmap_name: str) -> ListedColormap:
    """Return a colourmap with white at zero, then a ramp from *base_cmap_name*."""
    base = plt.cm.get_cmap(base_cmap_name)
    colors = ["white"] + [base(i) for i in np.linspace(0.2, 1.0, 255)]
    return ListedColormap(colors)


def add_frame_to_heatmap(ax) -> None:
    """Draw a bold black border around a heatmap axis."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_edgecolor("black")


# ═══════════════════════════════════════════════════════════════════════════
#  Three-panel contact map figure
# ═══════════════════════════════════════════════════════════════════════════


def plot_contact_maps(
    direct_contacts: np.ndarray,
    cla_matrix: np.ndarray,
    sod_matrix: np.ndarray,
    ph_dot: str,
    cutoff: float,
    fold: str,
    ph: str,
    zoom_range: Optional[Tuple[int, int]] = None,
    suffix: str = "full",
) -> None:
    """Three-panel heatmap: direct, Cl⁻-mediated, Na⁺-mediated.

    Parameters
    ----------
    zoom_range : tuple (start, end) or None
        If provided, crop both axes to the specified display-residue
        window (e.g. ``(263, 319)``).  The same window is applied to
        both chains.
    suffix : str
        Appended to the output filename (e.g. ``"full"`` or
        ``"zoom_263-319"``).
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    rpc = RESIDUES_PER_CHAIN  # shorthand

    # --- optional zoom ---------------------------------------------------
    if zoom_range is not None:
        zs, ze = zoom_range
        zs_idx = max(0, zs - DISPLAY_START_RESIDUE)
        ze_idx = min(rpc - 1, ze - DISPLAY_START_RESIDUE)

        # Indices for both chains
        zoom_idx = (
            list(range(zs_idx, ze_idx + 1))
            + list(range(rpc + zs_idx, rpc + ze_idx + 1))
        )

        direct_plot = direct_contacts[np.ix_(zoom_idx, zoom_idx)]
        cla_plot = cla_matrix[np.ix_(zoom_idx, zoom_idx)]
        sod_plot = sod_matrix[np.ix_(zoom_idx, zoom_idx)]

        zoom_rpc = ze_idx - zs_idx + 1
        zoom_start_idx = zs_idx

        print(f"  Zoom: residues {zs}–{ze}, "
              f"matrix {len(zoom_idx)}×{len(zoom_idx)}")
    else:
        direct_plot = direct_contacts
        cla_plot = cla_matrix
        sod_plot = sod_matrix
        zoom_rpc = rpc
        zoom_start_idx = 0

    # --- tick positions ---------------------------------------------------
    spacing = 5 if zoom_range else 10
    start_offset = 4 if zoom_range else 9
    tick_pos = np.arange(start_offset, zoom_rpc, spacing)

    tick_lbl = [
        str(DISPLAY_START_RESIDUE + zoom_start_idx + int(p))
        for p in tick_pos
    ]

    # Duplicate for chain B
    tick_pos_ab = list(tick_pos) + list(tick_pos + zoom_rpc)
    tick_lbl_ab = tick_lbl * 2

    # --- colourmaps -------------------------------------------------------
    red_cmap = create_custom_colormap("Reds")
    blue_cmap = create_custom_colormap("Blues")
    purple_cmap = create_custom_colormap("Purples")

    # --- scale maxima -----------------------------------------------------
    direct_max = max(int(np.max(direct_plot)), 1)
    cla_max = max(int(np.max(cla_plot)), 1)
    sod_max = max(int(np.max(sod_plot)), 1)

    direct_scale = min(1500, int(np.ceil(direct_max / 100) * 100))
    cla_scale = min(40, max(10, int(np.ceil(cla_max / 5) * 5)))
    sod_scale = min(10, max(5, int(np.ceil(sod_max / 2) * 2)))

    # --- panel specs ------------------------------------------------------
    panels = [
        (direct_plot, red_cmap,    direct_scale, "Direct Contacts",        "Direct Contacts"),
        (cla_plot,    blue_cmap,   cla_scale,    "Cl⁻-mediated Contacts",  "Cl⁻-mediated Contacts"),
        (sod_plot,    purple_cmap, sod_scale,    "Na⁺-mediated Contacts",  "Na⁺-mediated Contacts"),
    ]

    for ax, (mat, cmap, scale, title_base, cbar_lbl) in zip(axes, panels):
        n_ticks = max(2, scale // max(1, scale // 6))
        cbar_ticks = np.arange(0, scale + 1, max(1, scale // 5))

        sns.heatmap(
            mat, cmap=cmap, square=True,
            vmin=0, vmax=scale,
            cbar_kws={
                "shrink": 0.8, "aspect": 20,
                "label": cbar_lbl, "ticks": cbar_ticks,
            },
            ax=ax, xticklabels=False, yticklabels=False,
        )

        ax.set_xticks(tick_pos_ab)
        ax.set_xticklabels(tick_lbl_ab, rotation=45)
        ax.set_yticks(tick_pos_ab)
        ax.set_yticklabels(tick_lbl_ab, rotation=0)
        ax.set_xlabel("Residue Number")
        ax.set_ylabel("Residue Number")

        title = f"{title_base}\n(cutoff={cutoff}Å, pH={ph_dot})"
        if zoom_range:
            title += f"\nZoom: {zoom_range[0]}–{zoom_range[1]}"
        ax.set_title(title)
        ax.invert_yaxis()

        # Chain-boundary lines
        ax.axhline(y=zoom_rpc, color="black", linewidth=2,
                    linestyle="--", alpha=0.8)
        ax.axvline(x=zoom_rpc, color="black", linewidth=2,
                    linestyle="--", alpha=0.8)
        add_frame_to_heatmap(ax)

    plt.tight_layout()

    outfile = (
        f"FOLD{fold}_pH{ph}_combined_contact_maps_{cutoff}A_{suffix}.png"
    )
    plt.savefig(outfile, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  Saved contact maps → {outfile}")
    print(f"    Direct max: {np.max(direct_plot)}")
    print(f"    Cl⁻ max   : {np.max(cla_plot)}")
    print(f"    Na⁺ max   : {np.max(sod_plot)}")

    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
#  Main pipeline
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Iterate over all pH conditions and produce all outputs."""
    print("Dimer Contact Analysis Pipeline")
    print(f"cutoff={CUTOFF} Å, matrix_size={MATRIX_SIZE}, fold={FOLD}")

    if USE_LAST_N_FRAMES is not None:
        print(f"Frame mode: last {USE_LAST_N_FRAMES} frames only")
    else:
        print("Frame mode: all frames")

    for condition in PH_CONDITIONS:
        ph = condition["PH"]
        ph_dot = condition["PHdot"]

        print(f"\n{'=' * 60}")
        print(f"  pH {ph_dot}")
        print(f"{'=' * 60}")

        cla_file = (
            f"FOLD{FOLD}_pH{ph}_ion_residues_with_CLA_interactions_"
            f"{CUTOFF}A.txt"
        )
        sod_file = (
            f"FOLD{FOLD}_pH{ph}_ion_residues_with_SOD_interactions_"
            f"{CUTOFF}A.txt"
        )
        pdb_file = f"FOLD{FOLD}_pH{ph}_ion_1ns.pdb"

        # -- Load ion interactions ----------------------------------------
        cla_df = load_ion_interactions(
            cla_file, "CLA", last_n_frames=USE_LAST_N_FRAMES
        )
        sod_df = load_ion_interactions(
            sod_file, "SOD", last_n_frames=USE_LAST_N_FRAMES
        )

        # -- Ion-mediated contacts ----------------------------------------
        print("\nCl⁻ analysis:")
        cla_mediated = calculate_ion_contacts(cla_df)
        cla_direct = calculate_direct_ion_interactions(cla_df)

        print("\nNa⁺ analysis:")
        sod_mediated = calculate_ion_contacts(sod_df)
        sod_direct = calculate_direct_ion_interactions(sod_df)

        cla_matrix = contacts_to_matrix(cla_mediated)
        sod_matrix = contacts_to_matrix(sod_mediated)
        cla_array = ion_contacts_to_array(cla_direct)
        sod_array = ion_contacts_to_array(sod_direct)

        # -- Direct contacts from PDB ------------------------------------
        direct_matrix, time_matrix = process_pdb_single_pass(
            pdb_file, last_n_frames=USE_LAST_N_FRAMES
        )
        direct_counts = calculate_residue_contact_counts(direct_matrix)

        # -- Console summary ----------------------------------------------
        print(f"\nSummary (pH {ph_dot}):")
        print(f"  Direct contacts       : {np.sum(direct_matrix) // 2}")
        print(f"  Cl⁻-mediated contacts : {np.sum(cla_matrix) // 2}")
        print(f"  Na⁺-mediated contacts : {np.sum(sod_matrix) // 2}")
        print(f"  Residue–Cl⁻ contacts  : {np.sum(cla_array)}")
        print(f"  Residue–Na⁺ contacts  : {np.sum(sod_array)}")

        # Residue range check
        _print_residue_range_check(cla_df, sod_df)

        # Top residues
        _print_top_residues("Direct", direct_counts, 10)
        _print_top_residues("Cl⁻", cla_array, 5)
        _print_top_residues("Na⁺", sod_array, 5)

        # -- Figures ------------------------------------------------------
        print("\nGenerating FULL contact maps …")
        plot_contact_maps(
            direct_matrix, cla_matrix, sod_matrix,
            ph_dot, CUTOFF, FOLD, ph,
            zoom_range=None, suffix="full",
        )

        print("\nGenerating ZOOMED contact maps (263–319) …")
        plot_contact_maps(
            direct_matrix, cla_matrix, sod_matrix,
            ph_dot, CUTOFF, FOLD, ph,
            zoom_range=(263, 319), suffix="zoom_263-319",
        )

        # -- Raw data export for repository deposition --------------------
        print("\nExporting raw data files …")
        export_pairwise_contacts_csv(direct_matrix, "direct", FOLD, ph)
        export_pairwise_contacts_csv(cla_matrix, "CLA_mediated", FOLD, ph)
        export_pairwise_contacts_csv(sod_matrix, "SOD_mediated", FOLD, ph)
        export_per_residue_counts_csv(
            direct_counts, cla_array, sod_array, FOLD, ph
        )
        export_full_matrix_csv(direct_matrix, "direct", FOLD, ph)
        export_full_matrix_csv(cla_matrix, "CLA_mediated", FOLD, ph)
        export_full_matrix_csv(sod_matrix, "SOD_mediated", FOLD, ph)
        export_time_matrix_csv(time_matrix, FOLD, ph)

    print("\nDone.")


# ═══════════════════════════════════════════════════════════════════════════
#  Console helpers
# ═══════════════════════════════════════════════════════════════════════════


def _print_residue_range_check(
    cla_df: pd.DataFrame, sod_df: pd.DataFrame
) -> None:
    """Print a quick sanity check of residue ranges in the data."""
    all_res: list = []
    if len(cla_df) > 0:
        all_res.extend(int(r.split("_")[1]) for r in cla_df["residue_id"])
    if len(sod_df) > 0:
        all_res.extend(int(r.split("_")[1]) for r in sod_df["residue_id"])

    if all_res:
        print(f"  Data residue range   : {min(all_res)}–{max(all_res)} "
              f"(expected {DATA_START_RESIDUE}–{DATA_END_RESIDUE})")
        print(f"  Display residue range: "
              f"{DISPLAY_START_RESIDUE}–{DISPLAY_END_RESIDUE}")


def _print_top_residues(
    label: str, counts: np.ndarray, n: int
) -> None:
    """Print the top-*n* residues by contact count."""
    if np.sum(counts) == 0:
        return

    top = np.argsort(counts)[-n:][::-1]
    print(f"\n  Top {n} {label} residues:")
    for rank, idx in enumerate(top, start=1):
        if counts[idx] == 0:
            break
        chain = "A" if idx < RESIDUES_PER_CHAIN else "B"
        data_res = DATA_START_RESIDUE + (idx % RESIDUES_PER_CHAIN)
        disp_res = DISPLAY_START_RESIDUE + (idx % RESIDUES_PER_CHAIN)
        print(f"    {rank}. {chain}_{disp_res} "
              f"(data:{data_res}): {counts[idx]}")


# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()
