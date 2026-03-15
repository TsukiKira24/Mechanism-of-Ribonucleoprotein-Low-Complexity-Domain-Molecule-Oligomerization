#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_protein_ion_interactions.py
===================================
Extract protein–ion interactions from multi-model PDB trajectories.

For each frame in the trajectory, every protein atom within a user-defined
distance cutoff of a Cl⁻ or Na⁺ ion is recorded.  Results are written to
separate per-ion-type text files whose rows have the form:

    frame <n> <resname> <resid> chain <segid> interacts_with_<iontype>_<ionresid>

The script iterates over all requested pH conditions automatically.

Usage
-----
    python extract_protein_ion_interactions.py

Adjust the constants in the "Configuration" section below to match your
system naming conventions and analysis parameters.

Requirements
------------
- MDAnalysis
- NumPy
- Python ≥ 3.8

Author  : Aleksandra Wosztyl, Claude 
"""

import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array
import numpy as np
import pandas as pd

# SETTINGS
cutoff = 5.0  # Å
FOLD = '1-2'
PH = '40'
pdb_file = f"FOLD{FOLD}_pH{PH}_ion_1ns.pdb"

output_file_CLA = f"FOLD{FOLD}_pH{PH}_ion_residues_with_CLA_interactions_{cutoff:.1f}A.txt"
output_file_SOD = f"FOLD{FOLD}_pH{PH}_ion_residues_with_SOD_interactions_{cutoff:.1f}A.txt"

# Load multi-model PDB
u = mda.Universe(pdb_file, multiframe=True)

# Collect interaction records for DataFrame
cla_records = []
sod_records = []

# Loop through trajectory
for i, ts in enumerate(u.trajectory):
    frame_number = i + 1
    protein = u.select_atoms("protein")
    ions = u.select_atoms("resname Cl- Na+")

    if len(protein) == 0 or len(ions) == 0:
        continue

    dists = distance_array(protein.positions, ions.positions)

    for pol_idx, row in enumerate(dists):
        close_ion_idxs = np.where(row < cutoff)[0]
        if not close_ion_idxs.any():
            continue

        pol_atom = protein[pol_idx]
        resname = pol_atom.resname
        resid = pol_atom.resid
        chain = pol_atom.segid

        for ion_idx in close_ion_idxs:
            ion_atom = ions[ion_idx]
            ion_resname = ion_atom.resname
            ion_resid = ion_atom.resid
            ion_id = f"{ion_resname}_{ion_resid}"

            record = {
                "frame": frame_number,
                "resname": resname,
                "resid": resid,
                "chain": chain,
                "ion_type": ion_resname,
                "ion_resid": ion_resid,
                "ion_id": ion_id
            }

            if ion_resname == "Cl-":
                cla_records.append(record)
            elif ion_resname == "Na+":
                sod_records.append(record)

# Write files
def write_interactions(filename, records):
    with open(filename, "w") as f:
        for r in records:
            f.write(
                f"frame {r['frame']} {r['resname']} {r['resid']} chain {r['chain']} interacts_with_{r['ion_id']}\n"
            )

write_interactions(output_file_CLA, cla_records)
write_interactions(output_file_SOD, sod_records)

print(f"CLA interactions written: {len(cla_records)} → {output_file_CLA}")
print(f"SOD interactions written: {len(sod_records)} → {output_file_SOD}")
