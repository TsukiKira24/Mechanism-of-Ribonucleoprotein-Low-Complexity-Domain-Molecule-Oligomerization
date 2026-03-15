#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fibril_contact_map_multiple_chains.py
=======================
Compute and visualise inter- vs intra-chain residue–residue contacts from
a multi-chain PDB structure, displayed as a single "quadrant" contact map.

The script reads a static PDB (e.g. a fibril assembly such as PDB 6WQK),
builds a global residue ordering across all chains, and computes a binary
contact matrix where two residues are "in contact" if any pair of their
non-hydrogen atoms falls within a distance cutoff.  The resulting map is
colour-coded so that **intra-chain contacts** (same chain) and
**inter-chain contacts** (different chains) are visually separated, with
dashed lines marking chain boundaries.

Outputs
-------
    * ``*_quadrant_contact_map_*.png``         – publication-quality heatmap
    * ``*_interactions_*.txt``                  – human-readable contact list
      with intra/inter classification
    * ``*_simple_interactions_*.txt``           – compact residue-pair list
    * ``*_pairwise_contacts_*.csv``            – machine-readable CSV for
      repository deposition (one row per contact pair)
    * ``*_per_residue_contact_counts_*.csv``   – per-residue summary
    * ``*_full_contact_matrix_*.csv``          – complete N×N binary matrix

Usage
-----
    python Fibril_contact_map_multiple_chains.py

Or call programmatically::

    analyze_quadrant_contacts("6WQK.pdb", cutoff=5.0)

Requirements
------------
    NumPy, pandas, matplotlib

Author  : Aleksandra Wosztyl
"""
