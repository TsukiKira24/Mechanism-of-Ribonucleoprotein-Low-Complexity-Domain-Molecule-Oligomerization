# -*- coding: utf-8 -*-
"""
Improved Contact Analysis Pipeline for Molecular Dynamics Simulations

Features:
- Enhanced configuration management with validation
- Comprehensive error handling and logging
- Optimized memory usage and performance
- Progress tracking and timing information
- Robust file I/O with validation
- Compact professional visualization with consistent styling
- Detailed reporting and statistics

@author: Improved version with compact visualization
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from collections import defaultdict
import os
import re
import numpy as np
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
from contextlib import contextmanager

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

try:
    import MDAnalysis as mda
    from MDAnalysis.analysis.distances import distance_array
    MDA_AVAILABLE = True
except ImportError:
    MDA_AVAILABLE = False
    print("⚠️  Warning: MDAnalysis not available. Direct contact analysis will be disabled.")

# =============================================================================
# CONFIGURATION AND SETUP
# =============================================================================

class Config:
    """Centralized configuration management with validation"""
    
    def __init__(self):
        # Core parameters
        self.cutoff = 5.0  # distance cutoff in Å
        self.fold = '1-2'
        self.matrix_size = 154
        self.start_residue = 188
        self.end_residue = 341
        
        # pH conditions to process
        self.ph_conditions = [
            {'PH': '40', 'PHdot': '4.0'},
            {'PH': '74', 'PHdot': '7.4'},
             {'PH': '85', 'PHdot': '8.5'}
        ]
        
        # Compact visualization settings
        self.dpi = 300
        self.figsize_heatmap = (12, 4)  # Much more compact
        self.figsize_barplot = (8, 5)   # Smaller bar plots
        self.tick_interval = 20  # Less crowded ticks
        
        # Performance settings
        self.progress_interval = 10  # frames between progress updates
        self.memory_efficient = True
        
    def validate(self) -> bool:
        """Validate configuration parameters"""
        try:
            assert self.cutoff > 0, "Cutoff must be positive"
            assert self.matrix_size > 0, "Matrix size must be positive"
            assert self.start_residue > 0, "Start residue must be positive"
            assert self.end_residue >= self.start_residue, "End residue must be >= start residue"
            assert len(self.ph_conditions) > 0, "At least one pH condition required"
            
            expected_size = self.end_residue - self.start_residue + 1
            if expected_size != self.matrix_size:
                logging.warning(f"Matrix size mismatch: expected {expected_size}, got {self.matrix_size}")
            
            return True
        except AssertionError as e:
            logging.error(f"Configuration validation failed: {e}")
            return False

def setup_logging() -> logging.Logger:
    """Setup comprehensive logging"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler('contact_analysis.log')
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger

@contextmanager
def timer(description: str, logger: logging.Logger):
    """Context manager for timing operations"""
    start = time.time()
    logger.info(f"Starting: {description}")
    try:
        yield
    finally:
        elapsed = time.time() - start
        logger.info(f"Completed: {description} in {elapsed:.2f} seconds")

# =============================================================================
# DATA LOADING AND PROCESSING
# =============================================================================

class DataLoader:
    """Handles all data loading operations with robust error handling"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def load_ion_interactions(self, filename: str, expected_ion_type: str) -> pd.DataFrame:
        """Load ion interaction data from text file with validation"""
        filepath = Path(filename)
        
        if not filepath.exists():
            self.logger.error(f"File '{filename}' not found")
            return pd.DataFrame()
        
        self.logger.info(f"Reading {expected_ion_type} contact file: {filename}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
            
            if not lines:
                self.logger.warning(f"File {filename} is empty")
                return pd.DataFrame()
            
            data = []
            skipped_lines = 0
            
            for line_num, line in enumerate(lines, 1):
                # Enhanced regex to handle various ion interaction patterns
                patterns = [
                    r".+?_(\d+)\s+(\w+)\s+(\d+)\s+interacts_with_(Cl-|Na\+)_(\d+)",
                    r"(\d+)\s+(\w+)\s+(\d+)\s+.*?(Cl-|Na\+)_(\d+)",  # Alternative format
                ]
                
                match = None
                for pattern in patterns:
                    match = re.search(pattern, line)
                    if match:
                        break
                
                if match:
                    try:
                        frame, resn, resi, ion_type, ion_resi = match.groups()
                        
                        # Validate data types
                        frame = int(frame)
                        resi = int(resi)
                        ion_resi = int(ion_resi)
                        
                        # Validate ranges
                        if not (1 <= resi <= self.config.matrix_size):
                            continue
                        
                        residue_id = f"{resn}_{resi}"
                        ion_id = f"{ion_type}_{ion_resi}"
                        data.append((frame, residue_id, ion_id, ion_type, ion_resi))
                        
                    except (ValueError, IndexError) as e:
                        skipped_lines += 1
                        continue
                else:
                    skipped_lines += 1
            
            if skipped_lines > 0:
                self.logger.warning(f"Skipped {skipped_lines} malformed lines in {filename}")
            
            df = pd.DataFrame(data, columns=["frame", "residue_id", "ion_id", "ion_type", "ion_resi"])
            self.logger.info(f"Loaded {len(df)} {expected_ion_type} interactions")
            
            # Validate ion type consistency
            if not df.empty:
                actual_types = df['ion_type'].unique()
                expected_symbol = 'Cl-' if expected_ion_type == 'CLA' else 'Na+'
                if expected_symbol not in actual_types:
                    self.logger.warning(f"Expected {expected_symbol} but found {actual_types}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading {filename}: {e}")
            return pd.DataFrame()

class ContactCalculator:
    """Handles contact calculations with optimized algorithms"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def calculate_ion_contacts(self, df: pd.DataFrame) -> Dict[Tuple[str, str], int]:
        """Calculate ion-mediated contacts with optimized algorithm"""
        if df.empty:
            return {}
        
        contacts = defaultdict(int)
        total_frames = df['frame'].nunique()
        
        self.logger.info(f"Calculating ion-mediated contacts across {total_frames} frames")
        
        with timer("Ion contact calculation", self.logger):
            # Group by frame for efficiency
            for frame, frame_group in df.groupby("frame"):
                # Group by ion for efficiency
                for ion_id, ion_group in frame_group.groupby("ion_id"):
                    residues = sorted(ion_group['residue_id'].unique())
                    
                    if len(residues) < 2:
                        continue
                    
                    # Calculate all pairwise combinations
                    for r1, r2 in combinations(residues, 2):
                        key = tuple(sorted((r1, r2)))
                        contacts[key] += 1
        
        self.logger.info(f"Found {len(contacts)} unique residue pairs with ion-mediated contacts")
        return contacts
    
    def contacts_to_matrix(self, contact_dict: Dict[Tuple[str, str], int]) -> np.ndarray:
        """Convert contact dictionary to matrix with validation"""
        matrix = np.zeros((self.config.matrix_size, self.config.matrix_size), dtype=np.int32)
        
        invalid_contacts = 0
        for (r1, r2), count in contact_dict.items():
            try:
                # Extract residue indices
                i = int(r1.split('_')[1]) - 1
                j = int(r2.split('_')[1]) - 1
                
                # Validate indices
                if 0 <= i < self.config.matrix_size and 0 <= j < self.config.matrix_size:
                    matrix[i, j] = matrix[j, i] = count
                else:
                    invalid_contacts += 1
            except (ValueError, IndexError):
                invalid_contacts += 1
                continue
        
        if invalid_contacts > 0:
            self.logger.warning(f"Skipped {invalid_contacts} contacts with invalid indices")
        
        return matrix
    
    def process_pdb_single_pass(self, pdb_filename: str) -> Tuple[np.ndarray, np.ndarray]:
        """Process PDB file optimally for direct contacts and time analysis"""
        if not MDA_AVAILABLE:
            self.logger.error("MDAnalysis not available for PDB processing")
            return (np.zeros((self.config.matrix_size, self.config.matrix_size), dtype=np.int32),
                   np.zeros((self.config.matrix_size, 1), dtype=np.int8))
        
        filepath = Path(pdb_filename)
        if not filepath.exists():
            self.logger.error(f"PDB file '{pdb_filename}' not found")
            return (np.zeros((self.config.matrix_size, self.config.matrix_size), dtype=np.int32),
                   np.zeros((self.config.matrix_size, 1), dtype=np.int8))
        
        self.logger.info(f"Loading structure: {pdb_filename}")
        
        try:
            with timer("PDB processing", self.logger):
                u = mda.Universe(str(filepath), format='PDB')
                n_frames = len(u.trajectory)
                self.logger.info(f"Found {n_frames} frames")
                
                # Initialize matrices with appropriate dtypes
                contact_counts = np.zeros((self.config.matrix_size, self.config.matrix_size), dtype=np.int32)
                time_matrix = np.zeros((self.config.matrix_size, n_frames), dtype=np.int8)
                
                # Process frames
                for frame_idx, ts in enumerate(u.trajectory):
                    if frame_idx % self.config.progress_interval == 0:
                        progress = (frame_idx + 1) / n_frames * 100
                        self.logger.info(f"Processing frame {frame_idx + 1}/{n_frames} ({progress:.1f}%)")
                    
                    # Select protein atoms (excluding hydrogens)
                    selection = "protein and not name H* and resid 1:152"
                    protein_atoms = u.select_atoms(selection)
                    
                    if len(protein_atoms) == 0:
                        self.logger.warning(f"No atoms found in frame {frame_idx + 1}")
                        continue
                    
                    # Get positions and residue IDs
                    positions = protein_atoms.positions
                    residue_ids = protein_atoms.resids - 1  # Convert to 0-based
                    
                    # Calculate distance matrix efficiently
                    dist_matrix = distance_array(positions, positions)
                    contact_matrix = dist_matrix < self.config.cutoff
                    
                    # Convert to residue-level contacts
                    frame_contacts = np.zeros((self.config.matrix_size, self.config.matrix_size), dtype=bool)
                    residues_in_contact = set()
                    
                    # Process contacts efficiently
                    contact_indices = np.where(contact_matrix)
                    for i, j in zip(contact_indices[0], contact_indices[1]):
                        res_i, res_j = residue_ids[i], residue_ids[j]
                        
                        if (res_i != res_j and 
                            0 <= res_i < self.config.matrix_size and 
                            0 <= res_j < self.config.matrix_size):
                            frame_contacts[res_i, res_j] = True
                            frame_contacts[res_j, res_i] = True
                            residues_in_contact.update([res_i, res_j])
                    
                    # Update matrices
                    contact_counts += frame_contacts.astype(np.int32)
                    
                    for res in residues_in_contact:
                        time_matrix[res, frame_idx] = 1
                
                self.logger.info("PDB processing completed successfully")
                return contact_counts, time_matrix
                
        except Exception as e:
            self.logger.error(f"Error processing PDB file: {e}")
            return (np.zeros((self.config.matrix_size, self.config.matrix_size), dtype=np.int32),
                   np.zeros((self.config.matrix_size, 1), dtype=np.int8))

# =============================================================================
# VISUALIZATION
# =============================================================================

class Visualizer:
    """Professional compact visualization with consistent styling"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.setup_style()
    
    def setup_style(self):
        """Setup consistent compact visualization style"""
        plt.rcParams.update({
            'font.size': 8,        # Smaller font for compact layout
            'axes.titlesize': 10,  # Compact titles
            'axes.labelsize': 8,   # Compact labels
            'xtick.labelsize': 7,  # Smaller tick labels
            'ytick.labelsize': 7,
            'legend.fontsize': 7,
            'figure.titlesize': 12,
            'figure.dpi': 100,     # Standard DPI for display
            'savefig.dpi': 300     # High DPI for saving
        })
    
    def create_custom_colormap(self, base_cmap_name: str):
        """Create custom colormap with white for zero values"""
        base_cmap = plt.cm.get_cmap(base_cmap_name)
        colors = base_cmap(np.linspace(0, 1, 256))
        colors[0] = [1, 1, 1, 1]  # White for zero
        return plt.matplotlib.colors.ListedColormap(colors)
    
    def plot_contact_maps(self, direct_contacts: np.ndarray, cla_matrix: np.ndarray, 
                         sod_matrix: np.ndarray, ph_dot: str, ph: str) -> bool:
        """Generate compact contact map visualization matching reference layout"""
        try:
            with timer("Contact maps generation", self.logger):
                # Create figure with compact layout
                fig, axes = plt.subplots(1, 3, figsize=self.config.figsize_heatmap)
                
                # Calculate sparse tick positions for cleaner appearance
                n_ticks = 6  # Maximum number of ticks
                tick_step = max(1, self.config.matrix_size // n_ticks)
                tick_indices = list(range(0, self.config.matrix_size, tick_step))
                tick_labels = [str(self.config.start_residue + i) for i in tick_indices]
                
                # Custom colormaps
                red_cmap = self.create_custom_colormap('Reds')
                blue_cmap = self.create_custom_colormap('Blues')
                purple_cmap = self.create_custom_colormap('Purples')
                
                # Plot 1: Direct contacts
                max_direct = np.max(direct_contacts) if np.max(direct_contacts) > 0 else 1
                
                im1 = axes[0].imshow(direct_contacts, cmap=red_cmap, aspect='equal',
                                   vmin=0, vmax=max_direct, origin='lower')
                axes[0].set_title('Direct Contact Map\n(non-H atoms)', fontsize=9, pad=10)
                axes[0].set_xlabel('Residue Number', fontsize=8)
                axes[0].set_ylabel('Residue Number', fontsize=8)
                
                # Set ticks
                axes[0].set_xticks(tick_indices)
                axes[0].set_xticklabels(tick_labels, rotation=45, fontsize=7)
                axes[0].set_yticks(tick_indices)
                axes[0].set_yticklabels(tick_labels, fontsize=7)
                
                # Compact colorbar
                cbar1 = plt.colorbar(im1, ax=axes[0], shrink=0.6, aspect=15)
                cbar1.set_label('Direct Contacts', fontsize=7)
                cbar1.ax.tick_params(labelsize=6)
                
                # Plot 2: Cl⁻ contacts
                max_cla = np.max(cla_matrix) if np.max(cla_matrix) > 0 else 1
                
                im2 = axes[1].imshow(cla_matrix, cmap=blue_cmap, aspect='equal',
                                   vmin=0, vmax=max_cla, origin='lower')
                axes[1].set_title('Cl- mediated Contact Map', fontsize=9, pad=10)
                axes[1].set_xlabel('Residue Number', fontsize=8)
                axes[1].set_ylabel('Residue Number', fontsize=8)
                
                axes[1].set_xticks(tick_indices)
                axes[1].set_xticklabels(tick_labels, rotation=45, fontsize=7)
                axes[1].set_yticks(tick_indices)
                axes[1].set_yticklabels(tick_labels, fontsize=7)
                
                cbar2 = plt.colorbar(im2, ax=axes[1], shrink=0.6, aspect=15)
                cbar2.set_label('Cl-mediated Contacts', fontsize=7)
                cbar2.ax.tick_params(labelsize=6)
                
                # Plot 3: Na⁺ contacts
                max_sod = np.max(sod_matrix) if np.max(sod_matrix) > 0 else 1
                
                im3 = axes[2].imshow(sod_matrix, cmap=purple_cmap, aspect='equal',
                                   vmin=0, vmax=max_sod, origin='lower')
                axes[2].set_title('Na+ mediated Contact Map', fontsize=9, pad=10)
                axes[2].set_xlabel('Residue Number', fontsize=8)
                axes[2].set_ylabel('Residue Number', fontsize=8)
                
                axes[2].set_xticks(tick_indices)
                axes[2].set_xticklabels(tick_labels, rotation=45, fontsize=7)
                axes[2].set_yticks(tick_indices)
                axes[2].set_yticklabels(tick_labels, fontsize=7)
                
                cbar3 = plt.colorbar(im3, ax=axes[2], shrink=0.6, aspect=15)
                cbar3.set_label('Na-mediated Contacts', fontsize=7)
                cbar3.ax.tick_params(labelsize=6)
                
                # Adjust layout for compact appearance
                plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle
                plt.subplots_adjust(wspace=0.4)  # Adjust spacing between plots
                
                # Save with error handling
                output_file = f"FOLD{self.config.fold}_pH{ph}_combined_contact_maps_{self.config.cutoff}A.png"
                plt.savefig(output_file, dpi=self.config.dpi, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                self.logger.info(f"Combined contact maps saved: {output_file}")
                
                plt.close()
                return True
                
        except Exception as e:
            self.logger.error(f"Error generating contact maps: {e}")
            return False
    
    def save_individual_barplots(self, direct_counts: np.ndarray, cla_counts: np.ndarray, 
                                sod_counts: np.ndarray, ph_dot: str, ph: str) -> bool:
        """Generate compact individual bar plots"""
        try:
            with timer("Bar plots generation", self.logger):
                residue_numbers = range(self.config.start_residue, self.config.end_residue + 1)
                
                # Calculate sparse tick positions
                n_ticks = 8
                tick_step = max(1, len(residue_numbers) // n_ticks)
                x_tick_positions = residue_numbers[::tick_step]
                
                def create_barplot(data: np.ndarray, color: str, title: str, ylabel: str, 
                                 max_val: int, suffix: str) -> bool:
                    """Create compact individual bar plot with professional styling"""
                    try:
                        fig, ax = plt.subplots(figsize=self.config.figsize_barplot)
                        
                        bars = ax.bar(residue_numbers, data, color=color, alpha=0.7, 
                                    width=1.0, linewidth=0.5, edgecolor='darkgray')
                        
                        # Compact styling
                        ax.set_title(title, fontsize=12, pad=15)
                        ax.set_xlabel("Residue Number", fontsize=10)
                        ax.set_ylabel(ylabel, fontsize=10)
                        ax.set_xticks(x_tick_positions)
                        ax.set_xticklabels([str(x) for x in x_tick_positions], rotation=45, fontsize=8)
                        ax.tick_params(axis='both', which='major', labelsize=8)
                        
                        # Set appropriate y-axis limits and ticks
                        if max_val > 0:
                            y_max = max_val * 1.05  # Tighter margins
                            y_ticks = np.linspace(0, max_val, min(6, max_val + 1), dtype=int)
                            ax.set_ylim(0, y_max)
                            ax.set_yticks(y_ticks)
                        
                        ax.grid(axis='y', alpha=0.3, linewidth=0.5)
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
                        
                        # Add value labels only for top 3 residues to avoid clutter
                        if np.max(data) > 0:
                            top_indices = np.argsort(data)[-3:]  # Top 3 residues only
                            for idx in top_indices:
                                if data[idx] > 0:
                                    ax.text(residue_numbers[idx], data[idx] + max_val * 0.02,
                                           f'{data[idx]:.0f}', ha='center', va='bottom', 
                                           fontsize=7, fontweight='bold')
                        
                        plt.tight_layout(pad=2.0)
                        
                        # Save with descriptive filename
                        filename = f"FOLD{self.config.fold}_pH{ph}_{suffix}_contacts_{self.config.cutoff}A.png"
                        plt.savefig(filename, dpi=self.config.dpi, bbox_inches='tight',
                                  facecolor='white', edgecolor='none')
                        self.logger.info(f"Bar plot saved: {filename}")
                        plt.close()
                        return True
                        
                    except Exception as e:
                        self.logger.error(f"Error creating {suffix} bar plot: {e}")
                        return False
                
                # Generate all three plots with appropriate max values
                max_direct = int(np.max(direct_counts)) if np.max(direct_counts) > 0 else 1
                max_cla = int(np.max(cla_counts)) if np.max(cla_counts) > 0 else 1
                max_sod = int(np.max(sod_counts)) if np.max(sod_counts) > 0 else 1
                
                success = True
                success &= create_barplot(direct_counts, '#d62728',  # Consistent red
                                        f'Direct Contact Counts per Residue (pH={ph_dot})',
                                        'Number of Contacts', max_direct, 'direct')
                
                success &= create_barplot(cla_counts, '#1f77b4',  # Consistent blue
                                        f'Cl⁻-mediated Contact Counts per Residue (pH={ph_dot})',
                                        'Number of Contacts', max_cla, 'cla')
                
                success &= create_barplot(sod_counts, '#9467bd',  # Consistent purple
                                        f'Na⁺-mediated Contact Counts per Residue (pH={ph_dot})',
                                        'Number of Contacts', max_sod, 'sod')
                
                return success
                
        except Exception as e:
            self.logger.error(f"Error generating bar plots: {e}")
            return False

# =============================================================================
# ANALYSIS AND REPORTING
# =============================================================================

class AnalysisReporter:
    """Generate comprehensive analysis reports"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def calculate_residue_contact_counts(self, matrix: np.ndarray, 
                                       df: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Calculate contact counts per residue with validation"""
        if df is not None:
            # For ion-mediated contacts
            residue_counts = np.zeros(self.config.matrix_size)
            for _, row in df.iterrows():
                try:
                    resi = int(row['residue_id'].split('_')[1]) - 1
                    if 0 <= resi < self.config.matrix_size:
                        residue_counts[resi] += 1
                except (ValueError, IndexError):
                    continue
            return residue_counts
        else:
            # For direct contacts
            return np.sum(matrix, axis=1)
    
    def generate_statistics_report(self, direct_contacts: np.ndarray, cla_matrix: np.ndarray,
                                 sod_matrix: np.ndarray, direct_counts: np.ndarray,
                                 cla_counts: np.ndarray, sod_counts: np.ndarray,
                                 time_matrix: np.ndarray, ph_dot: str, ph: str) -> bool:
        """Generate comprehensive statistics report"""
        try:
            output_file = f"FOLD{self.config.fold}_pH{ph}_analysis_report.txt"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                # Header
                f.write("="*80 + "\n")
                f.write(f"CONTACT ANALYSIS REPORT - pH {ph_dot}\n")
                f.write("="*80 + "\n\n")
                
                # Configuration
                f.write("CONFIGURATION:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Distance cutoff: {self.config.cutoff} Å\n")
                f.write(f"Fold: {self.config.fold}\n")
                f.write(f"Residue range: {self.config.start_residue}-{self.config.end_residue}\n")
                f.write(f"Matrix size: {self.config.matrix_size}\n")
                f.write(f"Frames analyzed: {time_matrix.shape[1]}\n\n")
                
                # Overall statistics
                f.write("OVERALL STATISTICS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total direct contacts: {np.sum(direct_contacts):,}\n")
                f.write(f"Total Cl⁻-mediated contacts: {np.sum(cla_matrix):,}\n")
                f.write(f"Total Na⁺-mediated contacts: {np.sum(sod_matrix):,}\n")
                f.write(f"Unique residue pairs (direct): {np.count_nonzero(direct_contacts) // 2:,}\n")
                f.write(f"Unique residue pairs (Cl⁻): {np.count_nonzero(cla_matrix) // 2:,}\n")
                f.write(f"Unique residue pairs (Na⁺): {np.count_nonzero(sod_matrix) // 2:,}\n\n")
                
                # Contact density
                max_possible = self.config.matrix_size * (self.config.matrix_size - 1) // 2
                f.write("CONTACT DENSITY:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Direct contact density: {(np.count_nonzero(direct_contacts) // 2) / max_possible * 100:.2f}%\n")
                f.write(f"Cl⁻ contact density: {(np.count_nonzero(cla_matrix) // 2) / max_possible * 100:.2f}%\n")
                f.write(f"Na⁺ contact density: {(np.count_nonzero(sod_matrix) // 2) / max_possible * 100:.2f}%\n\n")
                
                # Top contacted residues
                def write_top_residues(counts: np.ndarray, contact_type: str, n_top: int = 40):
                    f.write(f"TOP {n_top} MOST CONTACTED RESIDUES ({contact_type}):\n")
                    f.write("-" * 40 + "\n")
                    
                    if np.sum(counts) > 0:
                        top_indices = np.argsort(counts)[-n_top:][::-1]
                        for i, res_idx in enumerate(top_indices):
                            if counts[res_idx] > 0:
                                residue_num = res_idx + self.config.start_residue
                                f.write(f"  {i+1:2d}. Residue {residue_num:3d}: {counts[res_idx]:6.0f} contacts\n")
                    else:
                        f.write("  No contacts found\n")
                    f.write("\n")
                
                write_top_residues(direct_counts, "Direct")
                write_top_residues(cla_counts, "Cl⁻-mediated")
                write_top_residues(sod_counts, "Na⁺-mediated")
                
                # Contact distribution statistics
                f.write("CONTACT DISTRIBUTION STATISTICS:\n")
                f.write("-" * 40 + "\n")
                
                for name, counts in [("Direct", direct_counts), ("Cl⁻", cla_counts), ("Na⁺", sod_counts)]:
                    active_counts = counts[counts > 0]
                    if len(active_counts) > 0:
                        f.write(f"{name} contacts:\n")
                        f.write(f"  Residues with contacts: {len(active_counts)}/{len(counts)} ({len(active_counts)/len(counts)*100:.1f}%)\n")
                        f.write(f"  Mean contacts per active residue: {np.mean(active_counts):.1f}\n")
                        f.write(f"  Std dev: {np.std(active_counts):.1f}\n")
                        f.write(f"  Median: {np.median(active_counts):.1f}\n")
                        f.write(f"  Max: {np.max(active_counts):.0f}\n")
                        f.write(f"  Min: {np.min(active_counts):.0f}\n\n")
                
                # Time-based analysis
                if time_matrix.shape[1] > 1:
                    f.write("TIME-BASED ANALYSIS:\n")
                    f.write("-" * 40 + "\n")
                    
                    # Calculate persistence (fraction of time each residue is in contact)
                    persistence = np.mean(time_matrix, axis=1)
                    active_residues = persistence > 0
                    
                    f.write(f"Residues involved in contacts: {np.sum(active_residues)}/{len(persistence)} ({np.sum(active_residues)/len(persistence)*100:.1f}%)\n")
                    
                    if np.sum(active_residues) > 0:
                        f.write(f"Average contact persistence: {np.mean(persistence[active_residues]):.3f}\n")
                        f.write(f"Most persistent residue: {np.argmax(persistence) + self.config.start_residue} ({np.max(persistence):.3f})\n")
                        
                        # Top persistent residues
                        f.write("\nTOP 10 MOST PERSISTENT RESIDUES:\n")
                        top_persistent = np.argsort(persistence)[-10:][::-1]
                        for i, res_idx in enumerate(top_persistent):
                            if persistence[res_idx] > 0:
                                residue_num = res_idx + self.config.start_residue
                                f.write(f"  {i+1:2d}. Residue {residue_num:3d}: {persistence[res_idx]:.3f} ({persistence[res_idx]*100:.1f}%)\n")
                    
                    f.write("\n")
                
                # Footer
                f.write("="*80 + "\n")
                f.write(f"Report generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*80 + "\n")
            
            self.logger.info(f"Analysis report saved: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating analysis report: {e}")
            return False

# =============================================================================
# MAIN PIPELINE
# =============================================================================

class ContactAnalysisPipeline:
    """Main pipeline orchestrating the entire analysis"""
    
    def __init__(self):
        self.config = Config()
        self.logger = setup_logging()
        
        # Validate configuration
        if not self.config.validate():
            raise ValueError("Invalid configuration")
        
        # Initialize components
        self.data_loader = DataLoader(self.config, self.logger)
        self.contact_calculator = ContactCalculator(self.config, self.logger)
        self.visualizer = Visualizer(self.config, self.logger)
        self.reporter = AnalysisReporter(self.config, self.logger)
    
    def process_ph_condition(self, ph_condition: Dict[str, str]) -> bool:
        """Process a single pH condition"""
        ph = ph_condition['PH']
        ph_dot = ph_condition['PHdot']
        
        self.logger.info(f"Processing pH {ph_dot} condition")
        
        try:
            with timer(f"pH {ph_dot} analysis", self.logger):
                # Define file paths
                cla_filename = f"FOLD{self.config.fold}_pH{ph}_ion_residues_with_CLA_interactions_{self.config.cutoff}A.txt"
                sod_filename = f"FOLD{self.config.fold}_pH{ph}_ion_residues_with_SOD_interactions_{self.config.cutoff}A.txt"
                pdb_filename = f"FOLD{self.config.fold}_pH{ph}_ion_1ns.pdb"
                
                # Load ion interaction data
                cla_df = self.data_loader.load_ion_interactions(cla_filename, "CLA")
                sod_df = self.data_loader.load_ion_interactions(sod_filename, "SOD")
                
                # Calculate ion-mediated contacts
                cla_contacts = self.contact_calculator.calculate_ion_contacts(cla_df)
                sod_contacts = self.contact_calculator.calculate_ion_contacts(sod_df)
                
                # Convert to matrices
                cla_matrix = self.contact_calculator.contacts_to_matrix(cla_contacts)
                sod_matrix = self.contact_calculator.contacts_to_matrix(sod_contacts)
                
                # Process PDB for direct contacts
                direct_contacts, time_matrix = self.contact_calculator.process_pdb_single_pass(pdb_filename)
                
                # Calculate residue contact counts
                direct_counts = self.reporter.calculate_residue_contact_counts(direct_contacts)
                cla_counts = self.reporter.calculate_residue_contact_counts(cla_matrix, cla_df)
                sod_counts = self.reporter.calculate_residue_contact_counts(sod_matrix, sod_df)
                
                # Generate visualizations
                self.logger.info("Generating visualizations...")
                viz_success = True
                viz_success &= self.visualizer.plot_contact_maps(direct_contacts, cla_matrix, sod_matrix, ph_dot, ph)
                viz_success &= self.visualizer.save_individual_barplots(direct_counts, cla_counts, sod_counts, ph_dot, ph)
                
                # Generate analysis report
                report_success = self.reporter.generate_statistics_report(
                    direct_contacts, cla_matrix, sod_matrix,
                    direct_counts, cla_counts, sod_counts,
                    time_matrix, ph_dot, ph
                )
                
                # Log summary statistics
                self.logger.info(f"pH {ph_dot} Summary:")
                self.logger.info(f"  Direct contacts: {np.sum(direct_contacts):,}")
                self.logger.info(f"  Cl⁻-mediated contacts: {np.sum(cla_matrix):,}")
                self.logger.info(f"  Na⁺-mediated contacts: {np.sum(sod_matrix):,}")
                self.logger.info(f"  Frames processed: {time_matrix.shape[1]:,}")
                
                return viz_success and report_success
                
        except Exception as e:
            self.logger.error(f"Error processing pH {ph_dot}: {e}")
            return False
    
    def run(self) -> bool:
        """Execute the complete analysis pipeline"""
        self.logger.info("Starting Contact Analysis Pipeline")
        self.logger.info(f"Configuration: cutoff={self.config.cutoff}Å, matrix_size={self.config.matrix_size}, fold={self.config.fold}")
        
        if not MDA_AVAILABLE:
            self.logger.warning("MDAnalysis not available - direct contact analysis will be skipped")
        
        overall_success = True
        
        with timer("Complete analysis pipeline", self.logger):
            for condition in self.config.ph_conditions:
                success = self.process_ph_condition(condition)
                overall_success &= success
                
                if not success:
                    self.logger.warning(f"Failed to process pH {condition['PHdot']}")
        
        if overall_success:
            self.logger.info("Analysis pipeline completed successfully!")
        else:
            self.logger.warning("Analysis pipeline completed with some errors")
        
        return overall_success

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_environment() -> bool:
    """Validate the runtime environment"""
    logger = logging.getLogger(__name__)
    
    # Check required packages
    required_packages = ['pandas', 'matplotlib', 'seaborn', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {missing_packages}")
        return False
    
    if not MDA_AVAILABLE:
        logger.warning("MDAnalysis not available - some features will be disabled")
    
    # Check write permissions
    try:
        test_file = "test_write_permission.tmp"
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        logger.info("Write permissions verified")
    except Exception as e:
        logger.error(f"No write permission in current directory: {e}")
        return False
    
    return True

def main():
    """Main entry point with comprehensive error handling"""
    # Setup logging first
    logger = setup_logging()
    
    try:
        # Validate environment
        if not validate_environment():
            logger.error("Environment validation failed")
            return False
        
        # Create and run pipeline
        pipeline = ContactAnalysisPipeline()
        success = pipeline.run()
        
        if success:
            logger.info("Contact analysis completed successfully!")
            print("\nAnalysis completed successfully!")
            print("Check the generated files:")
            print("   - Contact maps: *_contact_maps_*.png")
            print("   - Bar plots: *_contacts_*.png")
            print("   - Analysis reports: *_analysis_report.txt")
            print("   - Log file: contact_analysis.log")
        else:
            logger.error("Analysis completed with errors")
            print("\nAnalysis completed with errors. Check contact_analysis.log for details.")
        
        return success
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        print("\nAnalysis interrupted by user")
        return False
        
    except Exception as e:
        logger.error(f"Unexpected error in main: {e}", exc_info=True)
        print(f"\nUnexpected error: {e}")
        print("Check contact_analysis.log for detailed error information")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
