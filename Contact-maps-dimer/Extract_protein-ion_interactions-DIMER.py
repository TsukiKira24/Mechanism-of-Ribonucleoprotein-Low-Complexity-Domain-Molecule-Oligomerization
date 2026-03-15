#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract_dimer_ion_interactions.py
=================================
Extract protein–ion interactions from **dimer** multi-model PDB
trajectories (Fold 2).

This is the dimer counterpart of ``extract_protein_ion_interactions.py``
(which handles the monomer / Fold 1-2 system).  The analysis logic is
identical — the only difference is the fold identifier and the
corresponding input/output filenames.

For each frame in the trajectory, every protein atom within a distance
cutoff of a Cl⁻ or Na⁺ ion is recorded.  Results are written to two
files per ion type per pH condition:

    1. A **legacy text file** consumed by the downstream contact analysis
       pipeline::

           frame <n> <resname> <resid> chain <segid> interacts_with_<ion>_<ionresid>

    2. A **CSV file** for repository deposition with explicit headers::

           frame, resname, resid, chain, ion_type, ion_resid, ion_id

The script iterates over all requested pH conditions automatically.

Usage
-----
    python Extract_dimer_ion_interactions.py

Requirements
------------
    MDAnalysis, NumPy, pandas

Author  : Aleksandra Wosztyl (Rizo Lab, UT Southwestern Medical Center)
Created : 2026
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array


# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════

FOLD: str = "2"
PH_CONDITIONS: Tuple[str, ...] = ("40", "74", "85")
CUTOFF_ANGSTROM: float = 5.0  # distance threshold (Å)

# File-naming templates — {fold}, {ph}, and {cutoff} are interpolated at
# runtime.  Modify these if your naming convention changes.
PDB_TEMPLATE: str = "FOLD{fold}_pH{ph}_ion_1ns.pdb"
OUTPUT_TEMPLATE_CLA: str = (
    "FOLD{fold}_pH{ph}_ion_residues_with_CLA_interactions_{cutoff:.1f}A.txt"
)
OUTPUT_TEMPLATE_SOD: str = (
    "FOLD{fold}_pH{ph}_ion_residues_with_SOD_interactions_{cutoff:.1f}A.txt"
)
CSV_TEMPLATE_CLA: str = (
    "FOLD{fold}_pH{ph}_CLA_interactions_{cutoff:.1f}A.csv"
)
CSV_TEMPLATE_SOD: str = (
    "FOLD{fold}_pH{ph}_SOD_interactions_{cutoff:.1f}A.csv"
)

# MDAnalysis selection strings
PROTEIN_SEL: str = "protein"
ION_SEL: str = "resname Cl- Na+"
CLA_RESNAME: str = "Cl-"
SOD_RESNAME: str = "Na+"


# ═══════════════════════════════════════════════════════════════════════════
#  Core helpers
# ═══════════════════════════════════════════════════════════════════════════


def collect_interactions(
    universe: mda.Universe,
    cutoff: float,
) -> Tuple[List[Dict], List[Dict]]:
    """Scan every frame for protein–ion contacts within *cutoff* Å.

    Parameters
    ----------
    universe : MDAnalysis.Universe
        A Universe loaded with a multi-model PDB (or any trajectory format).
    cutoff : float
        Distance threshold in Ångströms for defining an interaction.

    Returns
    -------
    cla_records, sod_records : list[dict], list[dict]
        Each record contains the keys: ``frame``, ``resname``, ``resid``,
        ``chain``, ``ion_type``, ``ion_resid``, ``ion_id``.
    """
    cla_records: List[Dict] = []
    sod_records: List[Dict] = []

    for frame_idx, _ts in enumerate(universe.trajectory):
        frame_number = frame_idx + 1

        protein = universe.select_atoms(PROTEIN_SEL)
        ions = universe.select_atoms(ION_SEL)

        if protein.n_atoms == 0 or ions.n_atoms == 0:
            continue

        # All-vs-all distance matrix: shape (N_protein, N_ions)
        dists = distance_array(protein.positions, ions.positions)

        for atom_idx, row in enumerate(dists):
            # BUG FIX: use .size instead of .any().  np.where returns
            # integer indices and int(0) is falsy, so the original
            # .any() check silently skipped contacts when the only
            # nearby ion sat at array index 0.
            close_ion_idxs = np.where(row < cutoff)[0]
            if close_ion_idxs.size == 0:
                continue

            atom = protein[atom_idx]

            for ion_idx in close_ion_idxs:
                ion = ions[ion_idx]
                record = {
                    "frame": frame_number,
                    "resname": atom.resname,
                    "resid": atom.resid,
                    "chain": atom.segid,
                    "ion_type": ion.resname,
                    "ion_resid": ion.resid,
                    "ion_id": f"{ion.resname}_{ion.resid}",
                }

                if ion.resname == CLA_RESNAME:
                    cla_records.append(record)
                elif ion.resname == SOD_RESNAME:
                    sod_records.append(record)

    return cla_records, sod_records


def write_interactions(filepath: Path, records: List[Dict]) -> None:
    """Write interaction records to the legacy whitespace-delimited format.

    This format is expected by the downstream contact analysis pipeline.

    Parameters
    ----------
    filepath : Path
        Destination file path.
    records : list[dict]
        Interaction records produced by :func:`collect_interactions`.
    """
    with open(filepath, "w") as fh:
        for r in records:
            fh.write(
                f"frame {r['frame']} {r['resname']} {r['resid']} "
                f"chain {r['chain']} interacts_with_{r['ion_id']}\n"
            )


def export_interactions_csv(filepath: Path, records: List[Dict]) -> None:
    """Write interaction records as a CSV for repository deposition.

    Parameters
    ----------
    filepath : Path
        Destination CSV path.
    records : list[dict]
        Interaction records produced by :func:`collect_interactions`.
    """
    columns = ["frame", "resname", "resid", "chain",
               "ion_type", "ion_resid", "ion_id"]

    if not records:
        # Write headers only so the file is still valid CSV
        pd.DataFrame(columns=columns).to_csv(filepath, index=False)
        return

    pd.DataFrame(records, columns=columns).to_csv(filepath, index=False)


# ═══════════════════════════════════════════════════════════════════════════
#  Per-condition pipeline
# ═══════════════════════════════════════════════════════════════════════════


def run_single_condition(fold: str, ph: str, cutoff: float) -> None:
    """Execute the full extraction for one fold / pH condition.

    Produces four output files (two text + two CSV) — one pair for Cl⁻
    and one pair for Na⁺.

    Parameters
    ----------
    fold : str
        Fold identifier (``"2"`` for the dimer system).
    ph : str
        pH label used in filenames (e.g. ``"85"`` for pH 8.5).
    cutoff : float
        Distance cutoff in Å.
    """
    pdb_path = Path(PDB_TEMPLATE.format(fold=fold, ph=ph))

    if not pdb_path.exists():
        print(f"[SKIP] PDB not found: {pdb_path}")
        return

    print(f"\n{'=' * 60}")
    print(f"  Fold {fold} (dimer)  |  pH {ph}  |  cutoff {cutoff:.1f} Å")
    print(f"  Input: {pdb_path}")
    print(f"{'=' * 60}")

    universe = mda.Universe(str(pdb_path), multiframe=True)

    cla_records, sod_records = collect_interactions(universe, cutoff)

    # -- Legacy text files (consumed by downstream pipeline) ------------------
    out_cla_txt = Path(
        OUTPUT_TEMPLATE_CLA.format(fold=fold, ph=ph, cutoff=cutoff)
    )
    out_sod_txt = Path(
        OUTPUT_TEMPLATE_SOD.format(fold=fold, ph=ph, cutoff=cutoff)
    )
    write_interactions(out_cla_txt, cla_records)
    write_interactions(out_sod_txt, sod_records)

    # -- CSV files (for repository deposition) --------------------------------
    out_cla_csv = Path(
        CSV_TEMPLATE_CLA.format(fold=fold, ph=ph, cutoff=cutoff)
    )
    out_sod_csv = Path(
        CSV_TEMPLATE_SOD.format(fold=fold, ph=ph, cutoff=cutoff)
    )
    export_interactions_csv(out_cla_csv, cla_records)
    export_interactions_csv(out_sod_csv, sod_records)

    print(f"  CLA contacts: {len(cla_records):>8,}")
    print(f"    txt → {out_cla_txt}")
    print(f"    csv → {out_cla_csv}")
    print(f"  SOD contacts: {len(sod_records):>8,}")
    print(f"    txt → {out_sod_txt}")
    print(f"    csv → {out_sod_csv}")


# ═══════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Iterate over all pH conditions defined in ``PH_CONDITIONS``."""
    print(
        f"Dimer protein–ion interaction extraction  |  "
        f"Fold {FOLD}  |  cutoff = {CUTOFF_ANGSTROM:.1f} Å"
    )
    print(f"pH conditions: {', '.join(PH_CONDITIONS)}")

    for ph in PH_CONDITIONS:
        run_single_condition(FOLD, ph, CUTOFF_ANGSTROM)

    print("\nDone.")


if __name__ == "__main__":
    main()
