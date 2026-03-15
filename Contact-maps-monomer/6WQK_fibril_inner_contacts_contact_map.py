#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
6WQK_fibril_inner_contact.py
=============================
Overlay intra-fragment contacts from every chain of a multi-chain PDB
structure onto a single consensus contact map.

Given a PDB with *N* identical chains (e.g. a pentameric assembly such as
PDB 6WQK), this script:

    1. Parses atom coordinates for a user-defined fragment (default
       residues 263–319) from each chain.
    2. Computes a binary residue–residue contact matrix per chain using
       a distance cutoff on non-hydrogen atoms.
    3. Sums the per-chain matrices into an **overlay matrix** whose
       entries count how many chains share each contact (0 → N).
    4. Generates a publication-quality heatmap coloured by contact
       frequency across chains.
    5. Exports raw data CSV files suitable for repository deposition.

Outputs
-------
    * ``*_OVERLAID_fragment_*.png``  – heatmap
    * ``*_per_chain_pairwise_contacts.csv``
    * ``*_overlay_pairwise_contacts.csv``
    * ``*_per_residue_contact_frequency.csv``
    * ``*_per_chain_full_matrix.csv``
    * ``*_overlay_full_matrix.csv``

Usage
-----
    python overlaid_fragment_analysis.py

Or call programmatically::

    analyze_overlaid_fragment("6WQK.pdb", cutoff=5.0)

Requirements
------------
    NumPy, pandas, matplotlib

Author  : Alex (Rizo Lab, UT Southwestern Medical Center)
Created : 2026
"""

from __future__ import annotations

import logging
import re
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Suppress non-critical warnings
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════


class OverlaidFragmentConfig:
    """Centralised configuration for the overlaid fragment analysis."""

    def __init__(self) -> None:
        # -- Input structure ---------------------------------------------------
        self.pdb_filename: str = "6WQK.pdb"

        # -- Fragment of interest ----------------------------------------------
        self.fragment_start: int = 263
        self.fragment_end: int = 319

        # -- Analysis parameters -----------------------------------------------
        self.cutoff: float = 5.0             # distance threshold (Å)
        self.exclude_hydrogens: bool = True  # drop H atoms before distance calc

        # -- Visualisation -----------------------------------------------------
        self.dpi: int = 300
        self.figsize_heatmap: Tuple[int, int] = (5, 4)

    # .....................................................................

    def get_fragment_size(self) -> int:
        """Number of residues in the fragment."""
        return self.fragment_end - self.fragment_start + 1

    def validate(self) -> bool:
        """Return ``True`` if the configuration is usable."""
        try:
            assert self.cutoff > 0, "Cutoff must be positive"
            assert Path(self.pdb_filename).exists(), (
                f"PDB file {self.pdb_filename} not found"
            )
            assert self.fragment_start < self.fragment_end, (
                "Fragment start must be < end"
            )
            return True
        except AssertionError as exc:
            print(f"Configuration validation failed: {exc}")
            return False


# ═══════════════════════════════════════════════════════════════════════════
#  LOGGING / TIMING
# ═══════════════════════════════════════════════════════════════════════════


def setup_logging() -> logging.Logger:
    """Create a logger writing to console and a log file."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        console = logging.StreamHandler()
        console.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(console)

        fh = logging.FileHandler("overlaid_fragment_analysis.log")
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
        logger.addHandler(fh)

    return logger


@contextmanager
def timer(description: str, logger: logging.Logger):
    """Context manager that logs wall-clock time for a block of code."""
    start = time.time()
    logger.info("Starting: %s", description)
    try:
        yield
    finally:
        logger.info(
            "Completed: %s in %.2f s", description, time.time() - start
        )


# ═══════════════════════════════════════════════════════════════════════════
#  PDB PARSING & CONTACT CALCULATION
# ═══════════════════════════════════════════════════════════════════════════


class OverlaidFragmentProcessor:
    """Parse a multi-chain PDB and compute per-chain fragment contacts."""

    def __init__(
        self, config: OverlaidFragmentConfig, logger: logging.Logger
    ) -> None:
        self.config = config
        self.logger = logger
        self.residue_info: Dict[Tuple[str, int], str] = {}

    # -----------------------------------------------------------------
    #  PDB parsing
    # -----------------------------------------------------------------

    def parse_pdb(
        self, filename: str
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Read ATOM/HETATM records and group by chain.

        Only atoms within the fragment range are retained.  Hydrogens
        are optionally excluded (controlled by ``config.exclude_hydrogens``).

        Returns
        -------
        chain_positions : dict[str, ndarray]
            ``{chain_id: (N_atoms, 3)}`` coordinate arrays.
        info : dict
            Keys: ``chains``, ``chain_residue_ids``, ``residue_info``.
        """
        self.logger.info("Parsing PDB file: %s", filename)

        chain_atoms: Dict[str, List[Dict]] = {}
        chains: set = set()

        try:
            with open(filename, "r") as fh:
                for line in fh:
                    if not (
                        line.startswith("ATOM")
                        or line.startswith("HETATM")
                    ):
                        continue

                    # PDB fixed-width format requires at least 54 columns
                    if len(line) < 54:
                        continue

                    atom_name = line[12:16].strip()
                    chain_id = line[21:22].strip() or "A"
                    residue_name = line[17:20].strip()

                    # Residue number — may contain insertion codes
                    res_field = line[22:26].strip()
                    try:
                        residue_num = int(res_field)
                    except ValueError:
                        res_match = re.match(r"(\d+)", res_field)
                        if res_match:
                            residue_num = int(res_match.group(1))
                        else:
                            continue

                    # Keep only fragment residues
                    if not (
                        self.config.fragment_start
                        <= residue_num
                        <= self.config.fragment_end
                    ):
                        continue

                    # Optionally skip hydrogens
                    if (
                        self.config.exclude_hydrogens
                        and atom_name.startswith("H")
                    ):
                        continue

                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                    except ValueError:
                        continue

                    chains.add(chain_id)
                    chain_atoms.setdefault(chain_id, []).append({
                        "atom_name": atom_name,
                        "residue_num": residue_num,
                        "residue_name": residue_name,
                        "x": x, "y": y, "z": z,
                    })
                    self.residue_info[(chain_id, residue_num)] = residue_name

            self.logger.info(
                "Found %d chains: %s", len(chains), sorted(chains)
            )

            # Build per-chain coordinate and residue-ID arrays
            chain_positions: Dict[str, np.ndarray] = {}
            chain_residue_ids: Dict[str, np.ndarray] = {}

            for cid, atoms in chain_atoms.items():
                chain_positions[cid] = np.array(
                    [[a["x"], a["y"], a["z"]] for a in atoms]
                )
                chain_residue_ids[cid] = np.array(
                    [a["residue_num"] for a in atoms]
                )
                self.logger.info(
                    "Chain %s: %d atoms in fragment %d–%d",
                    cid, len(atoms),
                    self.config.fragment_start, self.config.fragment_end,
                )

            info = {
                "chains": sorted(chains),
                "chain_residue_ids": chain_residue_ids,
                "residue_info": self.residue_info,
            }
            return chain_positions, info

        except Exception as exc:
            self.logger.error("Error parsing PDB: %s", exc)
            return {}, {}

    # -----------------------------------------------------------------
    #  Per-chain contact matrices
    # -----------------------------------------------------------------

    def calculate_fragment_contacts(
        self,
        chain_positions: Dict[str, np.ndarray],
        chain_residue_ids: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Compute a binary contact matrix for each chain's fragment.

        Two residues are "in contact" if any pair of their (non-H) atoms
        falls within ``config.cutoff`` Å.
        """
        frag_size = self.config.get_fragment_size()
        chain_matrices: Dict[str, np.ndarray] = {}

        for cid in sorted(chain_positions):
            self.logger.info("Calculating contacts for chain %s …", cid)

            positions = chain_positions[cid]
            resids = chain_residue_ids[cid]

            mat = np.zeros((frag_size, frag_size), dtype=np.int32)

            # All-vs-all atom distances
            dist = np.linalg.norm(
                positions[:, None, :] - positions[None, :, :], axis=2
            )
            ci, cj = np.where(dist < self.config.cutoff)

            for a, b in zip(ci, cj):
                if a >= b:
                    continue  # upper triangle only avoids double-counting
                ri, rj = resids[a], resids[b]
                if ri == rj:
                    continue  # skip intra-residue contacts

                # Map PDB residue numbers to 0-based fragment indices
                ii = ri - self.config.fragment_start
                jj = rj - self.config.fragment_start

                if 0 <= ii < frag_size and 0 <= jj < frag_size:
                    mat[ii, jj] = 1
                    mat[jj, ii] = 1

            chain_matrices[cid] = mat
            self.logger.info(
                "Chain %s: %d unique contacts",
                cid, np.count_nonzero(mat) // 2,
            )

        return chain_matrices

    # -----------------------------------------------------------------
    #  Overlay (consensus) matrices
    # -----------------------------------------------------------------

    def create_overlaid_matrix(
        self, chain_matrices: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sum per-chain contact matrices into an overlay.

        Returns
        -------
        sum_matrix : ndarray
            Entry (i, j) counts how many chains share the contact.
        consensus_matrix : ndarray
            Binary — 1 if at least one chain has the contact.
        """
        frag_size = self.config.get_fragment_size()
        sum_mat = np.zeros((frag_size, frag_size), dtype=np.int32)
        cons_mat = np.zeros((frag_size, frag_size), dtype=np.int32)

        for mat in chain_matrices.values():
            sum_mat += mat
            cons_mat = np.maximum(cons_mat, mat)

        n = len(chain_matrices)
        self.logger.info(
            "Overlay: %d contacts in any chain, %d in all %d chains",
            np.count_nonzero(cons_mat) // 2,
            np.count_nonzero(sum_mat == n) // 2,
            n,
        )
        return sum_mat, cons_mat


# ═══════════════════════════════════════════════════════════════════════════
#  RAW DATA EXPORT  (for repository deposition)
# ═══════════════════════════════════════════════════════════════════════════


class RawDataExporter:
    """Export all computed contact data as machine-readable CSV files.

    Naming scheme::

        <pdb_stem>_fragment_<start>-<end>_<cutoff>_<descriptor>.csv

    Exported files
    --------------
    *_per_chain_pairwise_contacts.csv
        One row per non-zero contact per chain (upper triangle only).
        Columns: ``chain, residue_i, residue_j``.

    *_overlay_pairwise_contacts.csv
        One row per residue pair contacted in ≥ 1 chain.
        Columns: ``residue_i, residue_j, n_chains_with_contact``.

    *_per_residue_contact_frequency.csv
        Per-residue summary across chains.
        Columns: ``residue_number, <chain>_contacts …,
        total_contacts, n_chains_with_any_contact``.

    *_per_chain_full_matrix.csv
        Stacked per-chain binary matrices (chain label in first column).

    *_overlay_full_matrix.csv
        The N × N overlay (sum) matrix with residue-number headers.
    """

    def __init__(
        self, config: OverlaidFragmentConfig, logger: logging.Logger
    ) -> None:
        self.config = config
        self.logger = logger

        # Common filename prefix
        stem = Path(config.pdb_filename).stem
        cutoff_str = f"{config.cutoff:.1f}A".replace(".", "p")
        self._prefix = (
            f"{stem}_fragment_{config.fragment_start}-{config.fragment_end}"
            f"_{cutoff_str}"
        )

    def _residue_numbers(self) -> np.ndarray:
        """Fragment residue numbers in PDB numbering."""
        return np.arange(
            self.config.fragment_start, self.config.fragment_end + 1
        )

    # -----------------------------------------------------------------
    #  Per-chain pairwise contact list
    # -----------------------------------------------------------------

    def export_per_chain_pairwise(
        self, chain_matrices: Dict[str, np.ndarray]
    ) -> Path:
        """Write every non-zero contact from each chain (upper triangle)."""
        outpath = Path(f"{self._prefix}_per_chain_pairwise_contacts.csv")
        rows: List[Tuple] = []
        fs = self.config.fragment_start

        for cid in sorted(chain_matrices):
            mat = chain_matrices[cid]
            size = mat.shape[0]
            for i in range(size):
                for j in range(i + 1, size):
                    if mat[i, j] > 0:
                        rows.append((cid, fs + i, fs + j))

        df = pd.DataFrame(rows, columns=["chain", "residue_i", "residue_j"])
        df.to_csv(outpath, index=False)
        self.logger.info(
            "Exported %d per-chain pairwise contacts → %s", len(df), outpath
        )
        return outpath

    # -----------------------------------------------------------------
    #  Overlay pairwise contact list
    # -----------------------------------------------------------------

    def export_overlay_pairwise(
        self, sum_matrix: np.ndarray
    ) -> Path:
        """Write non-zero overlay entries (upper triangle)."""
        outpath = Path(f"{self._prefix}_overlay_pairwise_contacts.csv")
        rows: List[Tuple] = []
        fs = self.config.fragment_start
        size = sum_matrix.shape[0]

        for i in range(size):
            for j in range(i + 1, size):
                if sum_matrix[i, j] > 0:
                    rows.append((fs + i, fs + j, int(sum_matrix[i, j])))

        df = pd.DataFrame(
            rows,
            columns=["residue_i", "residue_j", "n_chains_with_contact"],
        )
        df.to_csv(outpath, index=False)
        self.logger.info(
            "Exported %d overlay pairwise contacts → %s", len(df), outpath
        )
        return outpath

    # -----------------------------------------------------------------
    #  Per-residue contact frequency
    # -----------------------------------------------------------------

    def export_per_residue_frequency(
        self, chain_matrices: Dict[str, np.ndarray]
    ) -> Path:
        """Per-residue contact count for each chain plus totals."""
        outpath = Path(f"{self._prefix}_per_residue_contact_frequency.csv")
        res_nums = self._residue_numbers()
        data: Dict[str, list] = {"residue_number": res_nums.tolist()}

        total = np.zeros(len(res_nums), dtype=int)
        any_chain = np.zeros(len(res_nums), dtype=int)

        for cid in sorted(chain_matrices):
            # Number of contacts each residue participates in
            counts = np.sum(chain_matrices[cid], axis=1)
            data[f"chain_{cid}_contacts"] = counts.astype(int).tolist()
            total += counts.astype(int)
            any_chain += (counts > 0).astype(int)

        data["total_contacts"] = total.tolist()
        data["n_chains_with_any_contact"] = any_chain.tolist()

        df = pd.DataFrame(data)
        df.to_csv(outpath, index=False)
        self.logger.info("Exported per-residue frequency → %s", outpath)
        return outpath

    # -----------------------------------------------------------------
    #  Full matrices
    # -----------------------------------------------------------------

    def export_per_chain_full_matrices(
        self, chain_matrices: Dict[str, np.ndarray]
    ) -> Path:
        """Stack all per-chain binary matrices into one CSV."""
        outpath = Path(f"{self._prefix}_per_chain_full_matrix.csv")
        res_nums = self._residue_numbers()
        frames: List[pd.DataFrame] = []

        for cid in sorted(chain_matrices):
            df = pd.DataFrame(
                chain_matrices[cid],
                index=res_nums,
                columns=res_nums,
            )
            df.insert(0, "chain", cid)
            df.index.name = "residue"
            frames.append(df)

        combined = pd.concat(frames)
        combined.to_csv(outpath)
        self.logger.info(
            "Exported per-chain full matrices → %s", outpath
        )
        return outpath

    def export_overlay_full_matrix(
        self, sum_matrix: np.ndarray
    ) -> Path:
        """Write the complete overlay (sum) matrix."""
        outpath = Path(f"{self._prefix}_overlay_full_matrix.csv")
        res_nums = self._residue_numbers()

        df = pd.DataFrame(sum_matrix, index=res_nums, columns=res_nums)
        df.index.name = "residue"
        df.to_csv(outpath)
        self.logger.info("Exported overlay full matrix → %s", outpath)
        return outpath


# ═══════════════════════════════════════════════════════════════════════════
#  VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════


class OverlaidFragmentVisualizer:
    """Publication-quality overlay heatmap."""

    def __init__(
        self, config: OverlaidFragmentConfig, logger: logging.Logger
    ) -> None:
        self.config = config
        self.logger = logger
        self._apply_style()

    def _apply_style(self) -> None:
        """Set matplotlib rc params for compact figures."""
        plt.rcParams.update({
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.titlesize": 12,
            "figure.dpi": 100,
            "savefig.dpi": 300,
        })

    @staticmethod
    def _white_zero_cmap(base_name: str):
        """Return a copy of *base_name* whose lowest bin is pure white."""
        base = plt.cm.get_cmap(base_name)
        colours = base(np.linspace(0, 1, 256))
        colours[0] = [1, 1, 1, 1]
        return plt.matplotlib.colors.ListedColormap(colours)

    # -----------------------------------------------------------------

    def plot_overlaid_contact_map(
        self,
        sum_matrix: np.ndarray,
        chain_matrices: Dict[str, np.ndarray],
    ) -> bool:
        """Heatmap coloured by contact frequency across chains."""
        try:
            with timer("Overlaid contact map visualisation", self.logger):
                fig, ax = plt.subplots(
                    1, 1, figsize=self.config.figsize_heatmap
                )

                cmap = self._white_zero_cmap("RdYlBu_r")
                n_chains = len(chain_matrices)

                im = ax.imshow(
                    sum_matrix, cmap=cmap, aspect="equal",
                    vmin=0, vmax=n_chains, origin="lower",
                )

                # Tick positions
                frag_size = self.config.get_fragment_size()
                n_ticks = min(10, frag_size)
                step = max(1, frag_size // n_ticks)
                tick_pos = list(range(0, frag_size, step))
                tick_lbl = [
                    str(self.config.fragment_start + i) for i in tick_pos
                ]

                ax.set_xticks(tick_pos)
                ax.set_xticklabels(tick_lbl, rotation=45, fontsize=12)
                ax.set_yticks(tick_pos)
                ax.set_yticklabels(tick_lbl, fontsize=12)
                ax.set_xlabel("Residue Number", fontsize=12)
                ax.set_ylabel("Residue Number", fontsize=12)

                cbar = plt.colorbar(im, ax=ax, shrink=0.6, aspect=15)
                cbar.set_label(
                    f"Number of Chains\nwith Contact (max={n_chains})",
                    fontsize=7,
                )
                cbar.ax.tick_params(labelsize=6)

                plt.tight_layout()

                # Build output filename
                stem = Path(self.config.pdb_filename).stem
                cutoff_str = (
                    f"{self.config.cutoff:.1f}A".replace(".", "p")
                )
                outfile = (
                    f"{stem}_OVERLAID_fragment_"
                    f"{self.config.fragment_start}-"
                    f"{self.config.fragment_end}_"
                    f"{n_chains}chains_{cutoff_str}.png"
                )
                plt.savefig(
                    outfile, dpi=self.config.dpi, bbox_inches="tight",
                    facecolor="white", edgecolor="none",
                )
                self.logger.info("Saved overlaid contact map → %s", outfile)

                plt.show()
                return True

        except Exception as exc:
            self.logger.error(
                "Error generating overlaid contact map: %s", exc
            )
            return False


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN ANALYSER
# ═══════════════════════════════════════════════════════════════════════════


class OverlaidFragmentAnalyzer:
    """Orchestrates parsing → contacts → overlay → export → plot."""

    def __init__(self, config: OverlaidFragmentConfig) -> None:
        self.config = config
        self.logger = setup_logging()
        self.processor = OverlaidFragmentProcessor(config, self.logger)
        self.visualizer = OverlaidFragmentVisualizer(config, self.logger)
        self.exporter = RawDataExporter(config, self.logger)

    # -----------------------------------------------------------------

    def run_analysis(self) -> bool:
        """Execute the full pipeline and return success status."""
        self.logger.info(
            "Starting overlaid fragment analysis (%d–%d)",
            self.config.fragment_start,
            self.config.fragment_end,
        )

        if not self.config.validate():
            self.logger.error("Configuration validation failed")
            return False

        try:
            # -- Parse PDB ----------------------------------------------------
            chain_positions, info = self.processor.parse_pdb(
                self.config.pdb_filename
            )
            if not chain_positions:
                self.logger.error("Failed to parse PDB or extract fragment")
                return False

            chains = info["chains"]
            self.logger.info("Processing %d chains: %s", len(chains), chains)

            # -- Per-chain contact matrices ------------------------------------
            chain_matrices = self.processor.calculate_fragment_contacts(
                chain_positions, info["chain_residue_ids"]
            )

            # -- Overlay ------------------------------------------------------
            sum_matrix, consensus_matrix = (
                self.processor.create_overlaid_matrix(chain_matrices)
            )

            # -- Raw data export ----------------------------------------------
            self.logger.info("Exporting raw data files …")
            self.exporter.export_per_chain_pairwise(chain_matrices)
            self.exporter.export_overlay_pairwise(sum_matrix)
            self.exporter.export_per_residue_frequency(chain_matrices)
            self.exporter.export_per_chain_full_matrices(chain_matrices)
            self.exporter.export_overlay_full_matrix(sum_matrix)

            # -- Visualisation ------------------------------------------------
            success = self.visualizer.plot_overlaid_contact_map(
                sum_matrix, chain_matrices
            )

            if success:
                self.logger.info(
                    "Overlaid fragment analysis completed successfully"
                )
                self._print_summary(
                    sum_matrix, chain_matrices, consensus_matrix
                )

            return success

        except Exception as exc:
            self.logger.error("Critical error in overlaid analysis: %s", exc)
            import traceback
            traceback.print_exc()
            return False

    # -----------------------------------------------------------------

    def _print_summary(
        self,
        sum_matrix: np.ndarray,
        chain_matrices: Dict[str, np.ndarray],
        consensus_matrix: np.ndarray,
    ) -> None:
        """Print a human-readable summary to the console."""
        n_chains = len(chain_matrices)

        print("\n" + "=" * 70)
        print("OVERLAID FRAGMENT ANALYSIS SUMMARY")
        print("=" * 70)
        print(f"Structure       : {self.config.pdb_filename}")
        print(
            f"Fragment        : {self.config.fragment_start}"
            f"–{self.config.fragment_end} "
            f"({self.config.get_fragment_size()} residues)"
        )
        print(f"Chains overlaid : {n_chains}")
        print(f"Distance cutoff : {self.config.cutoff} Å")
        print()

        print("Individual chain contacts:")
        for cid, mat in sorted(chain_matrices.items()):
            print(f"  Chain {cid}: {np.count_nonzero(mat) // 2} contacts")

        contacts_any = np.count_nonzero(consensus_matrix) // 2
        contacts_all = np.count_nonzero(sum_matrix == n_chains) // 2

        print()
        print("Overlay statistics:")
        print(f"  Contacts in ANY chain        : {contacts_any}")
        print(f"  Contacts in ALL {n_chains} chains    : {contacts_all}")
        print(
            f"  Conservation rate            : "
            f"{contacts_all / max(1, contacts_any) * 100:.1f}%"
        )

        # List the output files actually produced
        stem = Path(self.config.pdb_filename).stem
        cutoff_str = f"{self.config.cutoff:.1f}A".replace(".", "p")
        tag = (
            f"{stem}_fragment_{self.config.fragment_start}"
            f"-{self.config.fragment_end}_{cutoff_str}"
        )

        print()
        print("Output files:")
        print(
            f"  {stem}_OVERLAID_fragment_"
            f"{self.config.fragment_start}-{self.config.fragment_end}_"
            f"{n_chains}chains_{cutoff_str}.png"
        )
        print(f"  {tag}_per_chain_pairwise_contacts.csv")
        print(f"  {tag}_overlay_pairwise_contacts.csv")
        print(f"  {tag}_per_residue_contact_frequency.csv")
        print(f"  {tag}_per_chain_full_matrix.csv")
        print(f"  {tag}_overlay_full_matrix.csv")
        print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════
#  PUBLIC ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════


def analyze_overlaid_fragment(
    pdb_file: str = "6WQK.pdb",
    cutoff: float = 5.0,
    fragment_start: int = 263,
    fragment_end: int = 319,
) -> bool:
    """Convenience function to run the overlaid fragment analysis.

    Parameters
    ----------
    pdb_file : str
        Path to a multi-chain PDB file.
    cutoff : float
        Distance threshold in Å.
    fragment_start, fragment_end : int
        Residue range defining the fragment.
    """
    print("Running Overlaid Fragment Analysis")
    print(f"PDB file        : {pdb_file}")
    print(f"Fragment        : {fragment_start}–{fragment_end}")
    print(f"Distance cutoff : {cutoff} Å")
    print(f"Mode            : overlay all chains onto single contact map")
    print("=" * 50)

    if not Path(pdb_file).exists():
        print(f"Error: {pdb_file} not found!")
        return False

    config = OverlaidFragmentConfig()
    config.pdb_filename = pdb_file
    config.cutoff = cutoff
    config.fragment_start = fragment_start
    config.fragment_end = fragment_end

    analyzer = OverlaidFragmentAnalyzer(config)
    return analyzer.run_analysis()


# ═══════════════════════════════════════════════════════════════════════════
#  SCRIPT ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    try:
        print("Required packages: numpy, pandas, matplotlib\n")

        success = analyze_overlaid_fragment(cutoff=5.0)

        if success:
            print("\n✓ Overlaid fragment analysis completed successfully.")
            print("\nThe overlay heatmap shows contact frequency across chains:")
            print("  colour intensity = number of chains sharing that contact.")
            print("Contacts can be classified as:")
            print("  Universal     – present in all chains")
            print("  Common        – present in multiple chains")
            print("  Chain-specific – present in only one chain")
        else:
            print("\nAnalysis failed. Check console output for details.")

    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")

    except Exception as exc:
        print(f"\nUnexpected error: {exc}")
        import traceback
        traceback.print_exc()
        raise SystemExit(1)

    raise SystemExit(0 if success else 1)
