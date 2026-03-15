#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contact_maps_and_barplots_for_LCD_domain.py
============================
Protein–ion contact analysis pipeline for molecular dynamics trajectories.

This script quantifies three classes of residue–residue contacts from
multi-model PDB trajectories of a protein monomer at varying pH:

    1. **Direct contacts** – non-hydrogen protein atoms within a distance
       cutoff, computed frame-by-frame from atomic coordinates via
       MDAnalysis.
    2. **Cl⁻-mediated contacts** – residue pairs that simultaneously
       coordinate the same chloride ion within the cutoff (parsed from
       pre-computed interaction files produced by
       ``extract_protein_ion_interactions.py``).
    3. **Na⁺-mediated contacts** – analogous to (2) for sodium ions.

For each pH condition the pipeline produces:

    * Symmetric contact-count matrices (154 × 154, residues 188–341).
    * Per-residue contact-count profiles.
    * A time-resolved binary matrix indicating which residues participate
      in direct contacts at each frame.
    * Publication-quality heatmaps and bar plots.
    * A plain-text statistics report.
    * **Raw data files** (CSV) for repository deposition: pairwise contact
      lists, per-residue counts, and frame-level time matrices.

Pipeline inputs
---------------
For each pH condition (e.g. ``PH = '74'``):

    * ``FOLD<fold>_pH<PH>_ion_residues_with_CLA_interactions_<cutoff>A.txt``
    * ``FOLD<fold>_pH<PH>_ion_residues_with_SOD_interactions_<cutoff>A.txt``
    * ``FOLD<fold>_pH<PH>_ion_1ns.pdb``  (multi-model trajectory)

Usage
-----
    Contact_maps_and_barplots_for_LCD_domain.py

Adjust parameters in the ``Config`` class below.

Requirements
------------
    MDAnalysis, NumPy, pandas, matplotlib

Author  : Aleksandra Wosztyl
"""

from __future__ import annotations

import logging
import re
import time
import warnings
from collections import defaultdict
from contextlib import contextmanager
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
#  Optional dependency: MDAnalysis (required only for direct-contact calc)
# ---------------------------------------------------------------------------
try:
    import MDAnalysis as mda
    from MDAnalysis.analysis.distances import distance_array

    MDA_AVAILABLE = True
except ImportError:
    MDA_AVAILABLE = False
    print(
        "⚠️  MDAnalysis not available. "
        "Direct contact analysis from PDB files will be disabled."
    )

# Suppress non-critical warnings (e.g. matplotlib deprecation notices)
warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════


class Config:
    """Centralised, validatable configuration for the analysis pipeline.

    All tuneable parameters live here so that nothing is hard-coded
    deeper in the analysis logic.
    """

    def __init__(self) -> None:
        # -- System definition ------------------------------------------------
        self.fold: str = "1-2"
        self.matrix_size: int = 154          # number of residues in the monomer
        self.start_residue: int = 188        # first residue number (PDB numbering)
        self.end_residue: int = 341          # last  residue number (PDB numbering)

        # -- Analysis parameters ----------------------------------------------
        self.cutoff: float = 5.0             # distance threshold (Å)

        # -- pH conditions to process -----------------------------------------
        #    PH  : label used in filenames (no dot)
        #    PHdot : human-readable label for plots and reports
        self.ph_conditions: List[Dict[str, str]] = [
            {"PH": "40", "PHdot": "4.0"},
            #{"PH": "74", "PHdot": "7.4"},
            #{"PH": "85", "PHdot": "8.5"},
        ]

        # -- Visualisation settings -------------------------------------------
        self.dpi: int = 300
        self.figsize_heatmap: Tuple[int, int] = (12, 4)
        self.figsize_barplot: Tuple[int, int] = (8, 5)

        # -- Performance / progress -------------------------------------------
        self.progress_interval: int = 10     # frames between log updates

    # .....................................................................

    def validate(self) -> bool:
        """Return ``True`` if every parameter is internally consistent."""
        try:
            assert self.cutoff > 0, "Cutoff must be positive"
            assert self.matrix_size > 0, "Matrix size must be positive"
            assert self.start_residue > 0, "Start residue must be positive"
            assert self.end_residue >= self.start_residue, (
                "End residue must be >= start residue"
            )
            assert len(self.ph_conditions) > 0, (
                "At least one pH condition required"
            )

            expected = self.end_residue - self.start_residue + 1
            if expected != self.matrix_size:
                logging.warning(
                    "Matrix size mismatch: residue range implies %d, "
                    "but matrix_size is %d",
                    expected,
                    self.matrix_size,
                )
            return True

        except AssertionError as exc:
            logging.error("Configuration validation failed: %s", exc)
            return False


# ═══════════════════════════════════════════════════════════════════════════
#  LOGGING / TIMING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════


def setup_logging() -> logging.Logger:
    """Create a logger that writes to both the console and a log file."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # -- Console ----------------------------------------------------------
        console = logging.StreamHandler()
        console.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(console)

        # -- File -------------------------------------------------------------
        fh = logging.FileHandler("contact_analysis.log")
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
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════


class DataLoader:
    """Load and validate ion-interaction files produced by the extraction step."""

    def __init__(self, config: Config, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

    # .....................................................................

    def load_ion_interactions(
        self, filename: str, expected_ion_type: str
    ) -> pd.DataFrame:
        """Parse a whitespace-delimited interaction file into a DataFrame.

        Parameters
        ----------
        filename : str
            Path to the interaction file (one atom-level contact per line).
        expected_ion_type : str
            ``"CLA"`` or ``"SOD"`` – used only for log messages and sanity
            checks; the actual ion identity is read from each line.

        Returns
        -------
        pd.DataFrame
            Columns: ``frame``, ``residue_id``, ``ion_id``, ``ion_type``,
            ``ion_resi``.  Empty DataFrame on failure.
        """
        filepath = Path(filename)
        if not filepath.exists():
            self.logger.error("File not found: %s", filename)
            return pd.DataFrame()

        self.logger.info("Reading %s contact file: %s", expected_ion_type, filename)

        try:
            with open(filepath, "r", encoding="utf-8") as fh:
                lines = [ln.strip() for ln in fh if ln.strip()]

            if not lines:
                self.logger.warning("File %s is empty", filename)
                return pd.DataFrame()

            data: List[Tuple] = []
            skipped = 0

            # Two regex patterns cover the known output formats produced by
            # extract_protein_ion_interactions.py and legacy scripts.
            patterns = [
                re.compile(
                    r".+?_(\d+)\s+(\w+)\s+(\d+)\s+interacts_with_"
                    r"(Cl-|Na\+)_(\d+)"
                ),
                re.compile(
                    r"(\d+)\s+(\w+)\s+(\d+)\s+.*?(Cl-|Na\+)_(\d+)"
                ),
            ]

            for line in lines:
                match = None
                for pat in patterns:
                    match = pat.search(line)
                    if match:
                        break

                if not match:
                    skipped += 1
                    continue

                try:
                    frame_str, resn, resi_str, ion_type, ion_resi_str = (
                        match.groups()
                    )
                    frame = int(frame_str)
                    resi = int(resi_str)
                    ion_resi = int(ion_resi_str)

                    # Keep only residues within the monomer range
                    if not (1 <= resi <= self.config.matrix_size):
                        continue

                    data.append((
                        frame,
                        f"{resn}_{resi}",        # residue_id
                        f"{ion_type}_{ion_resi}", # ion_id
                        ion_type,
                        ion_resi,
                    ))

                except (ValueError, IndexError):
                    skipped += 1

            if skipped:
                self.logger.warning(
                    "Skipped %d malformed lines in %s", skipped, filename
                )

            df = pd.DataFrame(
                data,
                columns=[
                    "frame", "residue_id", "ion_id", "ion_type", "ion_resi"
                ],
            )
            self.logger.info(
                "Loaded %d %s interactions", len(df), expected_ion_type
            )

            # Quick sanity check on the ion species present
            if not df.empty:
                expected_symbol = "Cl-" if expected_ion_type == "CLA" else "Na+"
                found = df["ion_type"].unique()
                if expected_symbol not in found:
                    self.logger.warning(
                        "Expected %s but found %s", expected_symbol, found
                    )

            return df

        except Exception as exc:
            self.logger.error("Error loading %s: %s", filename, exc)
            return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════
#  CONTACT CALCULATION
# ═══════════════════════════════════════════════════════════════════════════


class ContactCalculator:
    """Compute direct and ion-mediated residue–residue contacts."""

    def __init__(self, config: Config, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

    # -----------------------------------------------------------------
    #  Ion-mediated contacts
    # -----------------------------------------------------------------

    def calculate_ion_contacts(
        self, df: pd.DataFrame
    ) -> Dict[Tuple[str, str], int]:
        """Identify residue pairs bridged by the same ion in the same frame.

        For every (frame, ion) group the unique contacting residues are
        enumerated and all pairwise combinations are counted.

        Returns
        -------
        dict
            ``{(residue_id_A, residue_id_B): count}`` with keys sorted
            alphabetically so each pair appears exactly once.
        """
        if df.empty:
            return {}

        contacts: Dict[Tuple[str, str], int] = defaultdict(int)
        n_frames = df["frame"].nunique()
        self.logger.info(
            "Calculating ion-mediated contacts across %d frames", n_frames
        )

        with timer("Ion contact calculation", self.logger):
            # Group first by frame, then by ion identity
            for _frame, frame_grp in df.groupby("frame"):
                for _ion, ion_grp in frame_grp.groupby("ion_id"):
                    residues = sorted(ion_grp["residue_id"].unique())
                    if len(residues) < 2:
                        continue
                    for r1, r2 in combinations(residues, 2):
                        contacts[tuple(sorted((r1, r2)))] += 1

        self.logger.info(
            "Found %d unique residue pairs with ion-mediated contacts",
            len(contacts),
        )
        return contacts

    # -----------------------------------------------------------------

    def contacts_to_matrix(
        self, contact_dict: Dict[Tuple[str, str], int]
    ) -> np.ndarray:
        """Convert a pairwise contact dict into a symmetric count matrix.

        Residue indices are extracted from the ``resname_resid`` keys and
        mapped to 0-based matrix positions.
        """
        mat = np.zeros(
            (self.config.matrix_size, self.config.matrix_size), dtype=np.int32
        )
        invalid = 0

        for (r1, r2), count in contact_dict.items():
            try:
                i = int(r1.split("_")[1]) - 1
                j = int(r2.split("_")[1]) - 1
                if (
                    0 <= i < self.config.matrix_size
                    and 0 <= j < self.config.matrix_size
                ):
                    mat[i, j] = mat[j, i] = count
                else:
                    invalid += 1
            except (ValueError, IndexError):
                invalid += 1

        if invalid:
            self.logger.warning(
                "Skipped %d contacts with out-of-range indices", invalid
            )
        return mat

    # -----------------------------------------------------------------
    #  Direct contacts from PDB trajectory
    # -----------------------------------------------------------------

    def process_pdb_single_pass(
        self, pdb_filename: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute direct residue–residue contacts from a multi-model PDB.

        A single pass through the trajectory yields two arrays:

        * ``contact_counts`` – symmetric (N×N) matrix counting how many
          frames each residue pair is in contact.
        * ``time_matrix`` – binary (N×F) matrix flagging which residues
          participate in *any* contact at each frame.

        Only non-hydrogen protein atoms with ``resid 1–152`` are
        considered (matching the structured region of the monomer).
        """
        if not MDA_AVAILABLE:
            self.logger.error("MDAnalysis required for PDB processing")
            return self._empty_matrices(n_frames=1)

        filepath = Path(pdb_filename)
        if not filepath.exists():
            self.logger.error("PDB file not found: %s", pdb_filename)
            return self._empty_matrices(n_frames=1)

        self.logger.info("Loading structure: %s", pdb_filename)

        try:
            with timer("PDB processing", self.logger):
                u = mda.Universe(str(filepath), format="PDB")
                n_frames = len(u.trajectory)
                self.logger.info("Trajectory contains %d frames", n_frames)

                contact_counts = np.zeros(
                    (self.config.matrix_size, self.config.matrix_size),
                    dtype=np.int32,
                )
                time_matrix = np.zeros(
                    (self.config.matrix_size, n_frames), dtype=np.int8
                )

                # Atom selection: non-H protein atoms in the structured core
                selection = "protein and not name H* and resid 1:152"

                for idx, _ts in enumerate(u.trajectory):
                    if idx % self.config.progress_interval == 0:
                        self.logger.info(
                            "Frame %d / %d (%.1f %%)",
                            idx + 1,
                            n_frames,
                            (idx + 1) / n_frames * 100,
                        )

                    atoms = u.select_atoms(selection)
                    if atoms.n_atoms == 0:
                        self.logger.warning(
                            "No atoms selected in frame %d", idx + 1
                        )
                        continue

                    positions = atoms.positions
                    resids = atoms.resids - 1  # convert to 0-based

                    # All-vs-all distance matrix → boolean contact matrix
                    dists = distance_array(positions, positions)
                    in_contact = dists < self.config.cutoff

                    # Collapse atom-level contacts to residue-level
                    frame_contacts = np.zeros(
                        (self.config.matrix_size, self.config.matrix_size),
                        dtype=bool,
                    )
                    active_residues: set = set()

                    ci, cj = np.where(in_contact)
                    for a, b in zip(ci, cj):
                        ri, rj = resids[a], resids[b]
                        if (
                            ri != rj
                            and 0 <= ri < self.config.matrix_size
                            and 0 <= rj < self.config.matrix_size
                        ):
                            frame_contacts[ri, rj] = True
                            frame_contacts[rj, ri] = True
                            active_residues.update((ri, rj))

                    contact_counts += frame_contacts.astype(np.int32)
                    for res in active_residues:
                        time_matrix[res, idx] = 1

                self.logger.info("PDB processing completed")
                return contact_counts, time_matrix

        except Exception as exc:
            self.logger.error("Error processing PDB: %s", exc)
            return self._empty_matrices(n_frames=1)

    # -----------------------------------------------------------------

    def _empty_matrices(
        self, n_frames: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return zero-filled fallback matrices on error."""
        return (
            np.zeros(
                (self.config.matrix_size, self.config.matrix_size),
                dtype=np.int32,
            ),
            np.zeros(
                (self.config.matrix_size, n_frames), dtype=np.int8
            ),
        )


# ═══════════════════════════════════════════════════════════════════════════
#  RAW DATA EXPORT  (for repository deposition)
# ═══════════════════════════════════════════════════════════════════════════


class RawDataExporter:
    """Export every computed quantity as machine-readable CSV files.

    Deposited files follow a consistent naming scheme::

        FOLD<fold>_pH<PH>_<descriptor>_<cutoff>A.csv

    File descriptions
    -----------------
    *_direct_pairwise_contacts.csv
        One row per residue pair with a non-zero direct contact count.
        Columns: ``residue_i, residue_j, contact_count``.

    *_CLA_pairwise_contacts.csv / *_SOD_pairwise_contacts.csv
        Same format for Cl⁻- and Na⁺-mediated contacts.

    *_per_residue_contact_counts.csv
        Columns: ``residue_number, direct_contacts, cla_contacts,
        sod_contacts``.

    *_direct_contact_time_matrix.csv
        Binary (0/1) matrix with residues as rows and frames as columns,
        indicating participation in at least one direct contact per frame.
    """

    def __init__(self, config: Config, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

    # -----------------------------------------------------------------
    #  Pairwise contact lists
    # -----------------------------------------------------------------

    def export_pairwise_contacts(
        self,
        matrix: np.ndarray,
        label: str,
        ph: str,
    ) -> Path:
        """Write non-zero entries of a symmetric contact matrix to CSV.

        Only the upper triangle is written (i < j) to avoid duplication.
        """
        outpath = Path(
            f"FOLD{self.config.fold}_pH{ph}_{label}_pairwise_contacts_"
            f"{self.config.cutoff:.1f}A.csv"
        )

        rows: List[Tuple[int, int, int]] = []
        for i in range(self.config.matrix_size):
            for j in range(i + 1, self.config.matrix_size):
                if matrix[i, j] > 0:
                    rows.append((
                        i + self.config.start_residue,
                        j + self.config.start_residue,
                        int(matrix[i, j]),
                    ))

        df = pd.DataFrame(
            rows, columns=["residue_i", "residue_j", "contact_count"]
        )
        df.to_csv(outpath, index=False)
        self.logger.info(
            "Exported %d %s pairwise contacts → %s", len(df), label, outpath
        )
        return outpath

    # -----------------------------------------------------------------
    #  Per-residue summary
    # -----------------------------------------------------------------

    def export_per_residue_counts(
        self,
        direct_counts: np.ndarray,
        cla_counts: np.ndarray,
        sod_counts: np.ndarray,
        ph: str,
    ) -> Path:
        """Write per-residue contact counts (all three contact types)."""
        outpath = Path(
            f"FOLD{self.config.fold}_pH{ph}_per_residue_contact_counts_"
            f"{self.config.cutoff:.1f}A.csv"
        )

        residue_numbers = np.arange(
            self.config.start_residue, self.config.end_residue + 1
        )

        df = pd.DataFrame({
            "residue_number": residue_numbers,
            "direct_contacts": direct_counts.astype(int),
            "cla_mediated_contacts": cla_counts.astype(int),
            "sod_mediated_contacts": sod_counts.astype(int),
        })
        df.to_csv(outpath, index=False)
        self.logger.info("Exported per-residue counts → %s", outpath)
        return outpath

    # -----------------------------------------------------------------
    #  Time-resolved binary matrix
    # -----------------------------------------------------------------

    def export_time_matrix(
        self,
        time_matrix: np.ndarray,
        ph: str,
    ) -> Path:
        """Write the binary residue × frame direct-contact matrix."""
        outpath = Path(
            f"FOLD{self.config.fold}_pH{ph}_direct_contact_time_matrix_"
            f"{self.config.cutoff:.1f}A.csv"
        )

        residue_numbers = np.arange(
            self.config.start_residue, self.config.end_residue + 1
        )
        frame_labels = [f"frame_{i + 1}" for i in range(time_matrix.shape[1])]

        df = pd.DataFrame(time_matrix, index=residue_numbers, columns=frame_labels)
        df.index.name = "residue_number"
        df.to_csv(outpath)
        self.logger.info("Exported time matrix → %s", outpath)
        return outpath

    # -----------------------------------------------------------------
    #  Full-matrix dump (for completeness / downstream tools)
    # -----------------------------------------------------------------

    def export_full_matrix(
        self,
        matrix: np.ndarray,
        label: str,
        ph: str,
    ) -> Path:
        """Write the complete N×N contact matrix as a CSV with residue headers."""
        outpath = Path(
            f"FOLD{self.config.fold}_pH{ph}_{label}_full_matrix_"
            f"{self.config.cutoff:.1f}A.csv"
        )

        residue_numbers = np.arange(
            self.config.start_residue, self.config.end_residue + 1
        )
        df = pd.DataFrame(
            matrix,
            index=residue_numbers,
            columns=residue_numbers,
        )
        df.index.name = "residue"
        df.to_csv(outpath)
        self.logger.info("Exported full %s matrix → %s", label, outpath)
        return outpath


# ═══════════════════════════════════════════════════════════════════════════
#  VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════


class Visualizer:
    """Publication-quality figures with compact, consistent styling."""

    def __init__(self, config: Config, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger
        self._apply_style()

    def _apply_style(self) -> None:
        """Set matplotlib rc params for compact, journal-ready figures."""
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

    # -----------------------------------------------------------------
    #  Helper: white-at-zero colourmap
    # -----------------------------------------------------------------

    @staticmethod
    def _white_zero_cmap(base_name: str):
        """Return a copy of *base_name* whose lowest bin is pure white."""
        base = plt.cm.get_cmap(base_name)
        colours = base(np.linspace(0, 1, 256))
        colours[0] = [1, 1, 1, 1]
        return plt.matplotlib.colors.ListedColormap(colours)

    # -----------------------------------------------------------------
    #  Contact-map heatmaps (1 × 3 panel)
    # -----------------------------------------------------------------

    def plot_contact_maps(
        self,
        direct: np.ndarray,
        cla: np.ndarray,
        sod: np.ndarray,
        ph_dot: str,
        ph: str,
    ) -> bool:
        """Three-panel heatmap: direct, Cl⁻-mediated, Na⁺-mediated."""
        try:
            with timer("Contact maps generation", self.logger):
                fig, axes = plt.subplots(1, 3, figsize=self.config.figsize_heatmap)

                # Sparse tick positions so axis labels stay legible
                n_ticks = 6
                step = max(1, self.config.matrix_size // n_ticks)
                tick_idx = list(range(0, self.config.matrix_size, step))
                tick_lbl = [
                    str(self.config.start_residue + i) for i in tick_idx
                ]

                # One panel per contact type
                panels = [
                    (direct, "Reds",    "Direct Contact Map\n(non-H atoms)", "Direct Contacts"),
                    (cla,    "Blues",   "Cl⁻-mediated Contact Map",          "Cl⁻ Contacts"),
                    (sod,    "Purples", "Na⁺-mediated Contact Map",          "Na⁺ Contacts"),
                ]

                for ax, (mat, cmap_name, title, cbar_label) in zip(axes, panels):
                    vmax = max(int(np.max(mat)), 1)
                    cmap = self._white_zero_cmap(cmap_name)

                    im = ax.imshow(
                        mat, cmap=cmap, aspect="equal",
                        vmin=0, vmax=vmax, origin="lower",
                    )
                    ax.set_title(title, fontsize=9, pad=10)
                    ax.set_xlabel("Residue Number", fontsize=8)
                    ax.set_ylabel("Residue Number", fontsize=8)
                    ax.set_xticks(tick_idx)
                    ax.set_xticklabels(tick_lbl, rotation=45, fontsize=7)
                    ax.set_yticks(tick_idx)
                    ax.set_yticklabels(tick_lbl, fontsize=7)

                    cbar = plt.colorbar(im, ax=ax, shrink=0.6, aspect=15)
                    cbar.set_label(cbar_label, fontsize=7)
                    cbar.ax.tick_params(labelsize=6)

                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plt.subplots_adjust(wspace=0.4)

                outfile = (
                    f"FOLD{self.config.fold}_pH{ph}_combined_contact_maps_"
                    f"{self.config.cutoff}A.png"
                )
                plt.savefig(
                    outfile, dpi=self.config.dpi, bbox_inches="tight",
                    facecolor="white", edgecolor="none",
                )
                plt.close()
                self.logger.info("Saved contact maps → %s", outfile)
                return True

        except Exception as exc:
            self.logger.error("Error generating contact maps: %s", exc)
            return False

    # -----------------------------------------------------------------
    #  Per-residue bar plots (one figure per contact type)
    # -----------------------------------------------------------------

    def save_individual_barplots(
        self,
        direct_counts: np.ndarray,
        cla_counts: np.ndarray,
        sod_counts: np.ndarray,
        ph_dot: str,
        ph: str,
    ) -> bool:
        """Three separate bar plots coloured by contact type."""
        try:
            with timer("Bar plots generation", self.logger):
                res_nums = list(
                    range(self.config.start_residue, self.config.end_residue + 1)
                )
                n_ticks = 8
                step = max(1, len(res_nums) // n_ticks)
                xtick_pos = res_nums[::step]

                specs = [
                    (direct_counts, "#d62728", "Direct",       "direct"),
                    (cla_counts,    "#1f77b4", "Cl⁻-mediated", "cla"),
                    (sod_counts,    "#9467bd", "Na⁺-mediated", "sod"),
                ]

                ok = True
                for data, colour, label, suffix in specs:
                    ok &= self._single_barplot(
                        data, res_nums, xtick_pos, colour,
                        f"{label} Contact Counts per Residue (pH {ph_dot})",
                        "Number of Contacts",
                        int(max(np.max(data), 1)),
                        suffix, ph,
                    )
                return ok

        except Exception as exc:
            self.logger.error("Error generating bar plots: %s", exc)
            return False

    # .....................................................................

    def _single_barplot(
        self,
        data: np.ndarray,
        x: list,
        xtick_pos: list,
        colour: str,
        title: str,
        ylabel: str,
        ymax: int,
        suffix: str,
        ph: str,
    ) -> bool:
        """Render and save one bar plot."""
        try:
            fig, ax = plt.subplots(figsize=self.config.figsize_barplot)
            ax.bar(
                x, data, color=colour, alpha=0.7,
                width=1.0, linewidth=0.5, edgecolor="darkgray",
            )

            ax.set_title(title, fontsize=12, pad=15)
            ax.set_xlabel("Residue Number", fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_xticks(xtick_pos)
            ax.set_xticklabels(
                [str(v) for v in xtick_pos], rotation=45, fontsize=8
            )
            ax.tick_params(axis="both", labelsize=8)

            if ymax > 0:
                ax.set_ylim(0, ymax * 1.05)
                ax.set_yticks(
                    np.linspace(0, ymax, min(6, ymax + 1), dtype=int)
                )

            ax.grid(axis="y", alpha=0.3, linewidth=0.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Annotate only the top-3 residues to keep the plot uncluttered
            if np.max(data) > 0:
                for idx in np.argsort(data)[-3:]:
                    if data[idx] > 0:
                        ax.text(
                            x[idx], data[idx] + ymax * 0.02,
                            f"{data[idx]:.0f}",
                            ha="center", va="bottom",
                            fontsize=7, fontweight="bold",
                        )

            plt.tight_layout(pad=2.0)

            fname = (
                f"FOLD{self.config.fold}_pH{ph}_{suffix}_contacts_"
                f"{self.config.cutoff}A.png"
            )
            plt.savefig(
                fname, dpi=self.config.dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none",
            )
            plt.close()
            self.logger.info("Saved bar plot → %s", fname)
            return True

        except Exception as exc:
            self.logger.error("Error in %s bar plot: %s", suffix, exc)
            return False


# ═══════════════════════════════════════════════════════════════════════════
#  STATISTICS REPORT
# ═══════════════════════════════════════════════════════════════════════════


class AnalysisReporter:
    """Write a human-readable summary of all computed statistics."""

    def __init__(self, config: Config, logger: logging.Logger) -> None:
        self.config = config
        self.logger = logger

    # -----------------------------------------------------------------

    def calculate_residue_contact_counts(
        self,
        matrix: np.ndarray,
        df: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        """Per-residue contact counts.

        For *direct* contacts the row-sum of the matrix is used.  For
        *ion-mediated* contacts the counts come from the interaction
        DataFrame (number of atom-level contacts per residue), which
        preserves the original counting convention.
        """
        if df is not None:
            counts = np.zeros(self.config.matrix_size)
            for _, row in df.iterrows():
                try:
                    resi = int(row["residue_id"].split("_")[1]) - 1
                    if 0 <= resi < self.config.matrix_size:
                        counts[resi] += 1
                except (ValueError, IndexError):
                    continue
            return counts

        # Direct contacts: simple row-sum of the symmetric matrix
        return np.sum(matrix, axis=1).astype(float)

    # -----------------------------------------------------------------

    def generate_statistics_report(
        self,
        direct_contacts: np.ndarray,
        cla_matrix: np.ndarray,
        sod_matrix: np.ndarray,
        direct_counts: np.ndarray,
        cla_counts: np.ndarray,
        sod_counts: np.ndarray,
        time_matrix: np.ndarray,
        ph_dot: str,
        ph: str,
    ) -> bool:
        """Write a plain-text report covering all key statistics."""
        try:
            outfile = (
                f"FOLD{self.config.fold}_pH{ph}_analysis_report.txt"
            )

            max_possible_pairs = (
                self.config.matrix_size * (self.config.matrix_size - 1) // 2
            )

            with open(outfile, "w", encoding="utf-8") as f:
                # -- Header ---------------------------------------------------
                f.write("=" * 80 + "\n")
                f.write(f"CONTACT ANALYSIS REPORT — pH {ph_dot}\n")
                f.write("=" * 80 + "\n\n")

                # -- Configuration --------------------------------------------
                f.write("CONFIGURATION\n")
                f.write("-" * 40 + "\n")
                f.write(f"Distance cutoff     : {self.config.cutoff} Å\n")
                f.write(f"Fold                : {self.config.fold}\n")
                f.write(f"Residue range       : "
                        f"{self.config.start_residue}–{self.config.end_residue}\n")
                f.write(f"Matrix size         : {self.config.matrix_size}\n")
                f.write(f"Frames analysed     : {time_matrix.shape[1]}\n\n")

                # -- Global counts --------------------------------------------
                f.write("OVERALL STATISTICS\n")
                f.write("-" * 40 + "\n")
                for label, mat in [
                    ("Direct",       direct_contacts),
                    ("Cl⁻-mediated", cla_matrix),
                    ("Na⁺-mediated", sod_matrix),
                ]:
                    total = int(np.sum(mat))
                    pairs = np.count_nonzero(mat) // 2
                    density = pairs / max_possible_pairs * 100
                    f.write(
                        f"{label:18s}  total={total:>10,}   "
                        f"pairs={pairs:>6,}   "
                        f"density={density:.2f}%\n"
                    )
                f.write("\n")

                # -- Top contacted residues -----------------------------------
                def _write_top(counts, name, n=40):
                    f.write(f"TOP {n} MOST CONTACTED RESIDUES ({name})\n")
                    f.write("-" * 40 + "\n")
                    if np.sum(counts) == 0:
                        f.write("  No contacts found\n\n")
                        return
                    for rank, idx in enumerate(
                        np.argsort(counts)[-n:][::-1], start=1
                    ):
                        if counts[idx] > 0:
                            resnum = idx + self.config.start_residue
                            f.write(
                                f"  {rank:3d}. Residue {resnum:3d}: "
                                f"{counts[idx]:6.0f} contacts\n"
                            )
                    f.write("\n")

                _write_top(direct_counts, "Direct")
                _write_top(cla_counts, "Cl⁻-mediated")
                _write_top(sod_counts, "Na⁺-mediated")

                # -- Distribution stats ---------------------------------------
                f.write("CONTACT DISTRIBUTION STATISTICS\n")
                f.write("-" * 40 + "\n")
                for name, counts in [
                    ("Direct", direct_counts),
                    ("Cl⁻",    cla_counts),
                    ("Na⁺",    sod_counts),
                ]:
                    active = counts[counts > 0]
                    if len(active) == 0:
                        continue
                    pct = len(active) / len(counts) * 100
                    f.write(
                        f"{name} contacts:\n"
                        f"  Active residues : {len(active)}/{len(counts)} "
                        f"({pct:.1f}%)\n"
                        f"  Mean  : {np.mean(active):.1f}\n"
                        f"  Std   : {np.std(active):.1f}\n"
                        f"  Median: {np.median(active):.1f}\n"
                        f"  Range : {np.min(active):.0f}–{np.max(active):.0f}\n\n"
                    )

                # -- Time-based analysis --------------------------------------
                if time_matrix.shape[1] > 1:
                    f.write("TIME-BASED ANALYSIS\n")
                    f.write("-" * 40 + "\n")
                    persistence = np.mean(time_matrix, axis=1)
                    n_active = int(np.sum(persistence > 0))
                    f.write(
                        f"Residues in contact : {n_active}/{len(persistence)} "
                        f"({n_active / len(persistence) * 100:.1f}%)\n"
                    )
                    if n_active:
                        f.write(
                            f"Mean persistence    : "
                            f"{np.mean(persistence[persistence > 0]):.3f}\n"
                        )
                        best = int(np.argmax(persistence))
                        f.write(
                            f"Most persistent     : residue "
                            f"{best + self.config.start_residue} "
                            f"({persistence[best]:.3f})\n\n"
                        )
                        f.write("TOP 10 MOST PERSISTENT RESIDUES\n")
                        for rank, idx in enumerate(
                            np.argsort(persistence)[-10:][::-1], start=1
                        ):
                            if persistence[idx] > 0:
                                f.write(
                                    f"  {rank:2d}. Residue "
                                    f"{idx + self.config.start_residue:3d}: "
                                    f"{persistence[idx]:.3f} "
                                    f"({persistence[idx]*100:.1f}%)\n"
                                )
                    f.write("\n")

                # -- Footer ---------------------------------------------------
                f.write("=" * 80 + "\n")
                f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n")

            self.logger.info("Saved report → %s", outfile)
            return True

        except Exception as exc:
            self.logger.error("Error writing report: %s", exc)
            return False


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════


class ContactAnalysisPipeline:
    """Orchestrates the full analysis across all pH conditions."""

    def __init__(self) -> None:
        self.config = Config()
        self.logger = setup_logging()

        if not self.config.validate():
            raise ValueError("Invalid configuration — see log for details")

        # Instantiate all pipeline components
        self.loader = DataLoader(self.config, self.logger)
        self.calculator = ContactCalculator(self.config, self.logger)
        self.visualizer = Visualizer(self.config, self.logger)
        self.reporter = AnalysisReporter(self.config, self.logger)
        self.exporter = RawDataExporter(self.config, self.logger)

    # -----------------------------------------------------------------

    def process_ph_condition(self, ph_condition: Dict[str, str]) -> bool:
        """Run every analysis step for a single pH value."""
        ph = ph_condition["PH"]
        ph_dot = ph_condition["PHdot"]

        self.logger.info("Processing pH %s", ph_dot)

        try:
            with timer(f"pH {ph_dot} analysis", self.logger):
                # -- File paths -----------------------------------------------
                cla_file = (
                    f"FOLD{self.config.fold}_pH{ph}_ion_residues_with_CLA_"
                    f"interactions_{self.config.cutoff}A.txt"
                )
                sod_file = (
                    f"FOLD{self.config.fold}_pH{ph}_ion_residues_with_SOD_"
                    f"interactions_{self.config.cutoff}A.txt"
                )
                pdb_file = (
                    f"FOLD{self.config.fold}_pH{ph}_ion_1ns.pdb"
                )

                # -- Load ion interaction data --------------------------------
                cla_df = self.loader.load_ion_interactions(cla_file, "CLA")
                sod_df = self.loader.load_ion_interactions(sod_file, "SOD")

                # -- Ion-mediated contact matrices ----------------------------
                cla_contacts = self.calculator.calculate_ion_contacts(cla_df)
                sod_contacts = self.calculator.calculate_ion_contacts(sod_df)
                cla_matrix = self.calculator.contacts_to_matrix(cla_contacts)
                sod_matrix = self.calculator.contacts_to_matrix(sod_contacts)

                # -- Direct contacts from PDB ---------------------------------
                direct_contacts, time_matrix = (
                    self.calculator.process_pdb_single_pass(pdb_file)
                )

                # -- Per-residue counts ---------------------------------------
                direct_counts = self.reporter.calculate_residue_contact_counts(
                    direct_contacts
                )
                cla_counts = self.reporter.calculate_residue_contact_counts(
                    cla_matrix, cla_df
                )
                sod_counts = self.reporter.calculate_residue_contact_counts(
                    sod_matrix, sod_df
                )

                # -- Figures --------------------------------------------------
                self.logger.info("Generating visualisations …")
                ok = True
                ok &= self.visualizer.plot_contact_maps(
                    direct_contacts, cla_matrix, sod_matrix, ph_dot, ph
                )
                ok &= self.visualizer.save_individual_barplots(
                    direct_counts, cla_counts, sod_counts, ph_dot, ph
                )

                # -- Text report ----------------------------------------------
                ok &= self.reporter.generate_statistics_report(
                    direct_contacts, cla_matrix, sod_matrix,
                    direct_counts, cla_counts, sod_counts,
                    time_matrix, ph_dot, ph,
                )

                # -- Raw data export for repository deposition ----------------
                self.logger.info("Exporting raw data files …")
                self.exporter.export_pairwise_contacts(
                    direct_contacts, "direct", ph
                )
                self.exporter.export_pairwise_contacts(cla_matrix, "CLA", ph)
                self.exporter.export_pairwise_contacts(sod_matrix, "SOD", ph)
                self.exporter.export_per_residue_counts(
                    direct_counts, cla_counts, sod_counts, ph
                )
                self.exporter.export_time_matrix(time_matrix, ph)

                # Full matrices (useful for downstream scripts)
                self.exporter.export_full_matrix(
                    direct_contacts, "direct", ph
                )
                self.exporter.export_full_matrix(cla_matrix, "CLA", ph)
                self.exporter.export_full_matrix(sod_matrix, "SOD", ph)

                # -- Console summary ------------------------------------------
                self.logger.info(
                    "pH %s summary — direct: %s | Cl⁻: %s | Na⁺: %s | "
                    "frames: %d",
                    ph_dot,
                    f"{np.sum(direct_contacts):,}",
                    f"{np.sum(cla_matrix):,}",
                    f"{np.sum(sod_matrix):,}",
                    time_matrix.shape[1],
                )
                return ok

        except Exception as exc:
            self.logger.error("Error processing pH %s: %s", ph_dot, exc)
            return False

    # -----------------------------------------------------------------

    def run(self) -> bool:
        """Iterate over all pH conditions and return overall success."""
        self.logger.info(
            "Contact Analysis Pipeline — fold=%s, cutoff=%.1f Å",
            self.config.fold,
            self.config.cutoff,
        )

        if not MDA_AVAILABLE:
            self.logger.warning(
                "MDAnalysis unavailable — direct contacts will be skipped"
            )

        ok = True
        with timer("Complete pipeline", self.logger):
            for cond in self.config.ph_conditions:
                if not self.process_ph_condition(cond):
                    self.logger.warning("Failed: pH %s", cond["PHdot"])
                    ok = False

        status = "successfully" if ok else "with errors"
        self.logger.info("Pipeline completed %s", status)
        return ok


# ═══════════════════════════════════════════════════════════════════════════
#  ENVIRONMENT CHECK / ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════


def validate_environment() -> bool:
    """Verify that required packages and write permissions are available."""
    logger = logging.getLogger(__name__)

    for pkg in ("pandas", "matplotlib", "numpy"):
        try:
            __import__(pkg)
        except ImportError:
            logger.error("Missing required package: %s", pkg)
            return False

    if not MDA_AVAILABLE:
        logger.warning("MDAnalysis missing — some analyses disabled")

    # Quick write-permission check
    try:
        tmp = Path("_write_test.tmp")
        tmp.write_text("ok")
        tmp.unlink()
    except Exception as exc:
        logger.error("Cannot write to working directory: %s", exc)
        return False

    return True


def main() -> bool:
    """Entry point with top-level error handling."""
    logger = setup_logging()

    try:
        if not validate_environment():
            logger.error("Environment validation failed")
            return False

        pipeline = ContactAnalysisPipeline()
        success = pipeline.run()

        if success:
            print("\n✓ Analysis completed successfully.")
            print("  Outputs: *_contact_maps_*.png, *_contacts_*.png,")
            print("           *_analysis_report.txt, *_pairwise_contacts.csv,")
            print("           *_per_residue_contact_counts.csv,")
            print("           *_direct_contact_time_matrix.csv,")
            print("           *_full_matrix.csv")
        else:
            print("\n⚠ Completed with errors — see contact_analysis.log")

        return success

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return False

    except Exception as exc:
        logger.error("Unexpected error: %s", exc, exc_info=True)
        return False


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
