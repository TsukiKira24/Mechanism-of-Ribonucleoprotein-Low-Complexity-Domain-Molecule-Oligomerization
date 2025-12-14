import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
from contextlib import contextmanager
import re

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

try:
    import MDAnalysis as mda
    from MDAnalysis.analysis.distances import distance_array
    MDA_AVAILABLE = True
except ImportError:
    MDA_AVAILABLE = False

class FibrilConfig:
    """Configuration for fibril contact analysis"""
    
    def __init__(self):
        self.cutoff = 5.0
        self.pdb_filename = "6WQK.pdb"
        self.dpi = 300
        self.figsize_heatmap = (12, 10)
        self.exclude_hydrogens = True
        
    def validate(self) -> bool:
        """Validate configuration parameters"""
        try:
            assert self.cutoff > 0, "Cutoff must be positive"
            assert Path(self.pdb_filename).exists(), f"PDB file {self.pdb_filename} not found"
            return True
        except AssertionError as e:
            print(f"Configuration validation failed: {e}")
            return False

def setup_logging() -> logging.Logger:
    """Setup logging for fibril analysis"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
    
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

class FibrilPDBProcessor:
    """Process PDB files for fibril contact analysis"""
    
    def __init__(self, config: FibrilConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.residue_info = {}
        self.chain_info = {}
        self.atom_chain_mapping = {}
    
    def robust_pdb_parser(self, filename: str) -> Tuple[np.ndarray, Dict, Dict]:
        """Robust PDB parser that handles various PDB formats"""
        self.logger.info(f"Using robust PDB parser for {filename}")
        
        atoms = []
        chains = set()
        residues = set()
        
        try:
            with open(filename, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        # Only process ATOM and HETATM records
                        if not (line.startswith('ATOM') or line.startswith('HETATM')):
                            continue
                        
                        # Skip if line is too short
                        if len(line) < 54:
                            continue
                        
                        # Parse fields with error handling
                        atom_name = line[12:16].strip()
                        chain_id = line[21:22].strip()
                        residue_name = line[17:20].strip()
                        
                        # Handle residue number parsing more robustly
                        res_field = line[22:26].strip()
                        try:
                            residue_num = int(res_field)
                        except ValueError:
                            # Handle insertion codes or alternative formats
                            res_match = re.match(r'(\d+)', res_field)
                            if res_match:
                                residue_num = int(res_match.group(1))
                            else:
                                self.logger.warning(f"Could not parse residue number '{res_field}' at line {line_num}")
                                continue
                        
                        # Skip hydrogens if configured
                        if self.config.exclude_hydrogens and atom_name.startswith('H'):
                            continue
                        
                        # Parse coordinates
                        try:
                            x = float(line[30:38].strip())
                            y = float(line[38:46].strip())
                            z = float(line[46:54].strip())
                        except ValueError:
                            self.logger.warning(f"Could not parse coordinates at line {line_num}")
                            continue
                        
                        atoms.append({
                            'atom_name': atom_name,
                            'chain_id': chain_id if chain_id else 'A',
                            'residue_num': residue_num,
                            'residue_name': residue_name,
                            'x': x, 'y': y, 'z': z,
                            'atom_index': len(atoms)
                        })
                        
                        chains.add(chain_id if chain_id else 'A')
                        residues.add(residue_num)
                        
                    except Exception as e:
                        self.logger.warning(f"Error parsing line {line_num}: {e}")
                        continue
            
            if not atoms:
                self.logger.error("No valid atoms found in PDB file")
                return np.array([]), {}, {}
            
            self.logger.info(f"Successfully parsed {len(atoms)} atoms from {len(chains)} chains")
            
            # Convert to arrays for processing
            positions = np.array([[atom['x'], atom['y'], atom['z']] for atom in atoms])
            chain_ids = np.array([atom['chain_id'] for atom in atoms])
            residue_keys = np.array([(atom['chain_id'], atom['residue_num']) for atom in atoms], dtype=object)

            # Build residue and chain info
            self.residue_info = {}
            for atom in atoms:
                key = (atom['chain_id'], atom['residue_num'])
                self.residue_info[key] = atom['residue_name']
            
            self.chain_info = {chain: [] for chain in chains}
            self.atom_chain_mapping = {}
            
            for i, atom in enumerate(atoms):
                if atom['residue_num'] not in self.chain_info[atom['chain_id']]:
                    self.chain_info[atom['chain_id']].append(atom['residue_num'])
                self.atom_chain_mapping[i] = atom['chain_id']
            
            # Sort residue lists
            for chain in self.chain_info:
                self.chain_info[chain].sort()
            
            return positions, residue_keys, chain_ids
            
        except Exception as e:
            self.logger.error(f"Error parsing PDB file: {e}")
            return np.array([]), {}, {}
    
    def analyze_structure_with_chain_separation(self, filename: str) -> Tuple[np.ndarray, Dict]:
        """Analyze structure and create chain-separated contact matrix"""

        positions, residue_keys, chain_ids = self.robust_pdb_parser(filename)
        if len(positions) == 0:
            return np.array([]), {}

        # Ordered list of chains
        chain_list = sorted(self.chain_info.keys())

        # Build global residue order as (chain, residue_num) pairs and boundaries per chain
        global_residue_order: List[Tuple[str, int]] = []
        chain_boundaries: Dict[str, Tuple[int, int]] = {}
        current_pos = 0
        for chain in chain_list:
            chain_residues = sorted(self.chain_info[chain])
            start = current_pos
            for r in chain_residues:
                global_residue_order.append((chain, r))
                current_pos += 1
            chain_boundaries[chain] = (start, current_pos)

        matrix_size = len(global_residue_order)
        contact_matrix = np.zeros((matrix_size, matrix_size), dtype=np.int32)

        # Map (chain, residue) -> matrix index
        residue_to_global_idx = {cr: idx for idx, cr in enumerate(global_residue_order)}

        self.logger.info(f"Calculating distances for {len(positions)} atoms...")

        # Distance matrix
        dist_matrix = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
        atom_contacts = dist_matrix < self.config.cutoff

        contacts_list = [] 
        
        # Fill residue-level contact matrix using composite keys
        ci, cj = np.where(atom_contacts)
        for i, j in zip(ci, cj):
            if i >= j:  # skip self/duplicate
                continue
            key_i = tuple(residue_keys[i])
            key_j = tuple(residue_keys[j])

            if key_i == key_j:
                continue  # same residue, skip

            if key_i in residue_to_global_idx and key_j in residue_to_global_idx:
                gi = residue_to_global_idx[key_i]
                gj = residue_to_global_idx[key_j]
                contact_matrix[gi, gj] = 1
                contact_matrix[gj, gi] = 1  # symmetric

                contacts_list.append((key_i, key_j))
        
        analysis_info = {
            'chains': chain_list,
            'chain_boundaries': chain_boundaries,
            'global_residue_order': global_residue_order,
            'matrix_size': matrix_size,
            'chain_info': self.chain_info,
            'residue_info': self.residue_info,
            'contacts_list': contacts_list
        }
        
        # Write single text output file
        self._write_contact_interactions(contacts_list, filename)
        
        print(f"\nTotal contacts found: {len(contacts_list)}")
        for c in contacts_list[:10]:
            res1_name = self.residue_info.get(c[0], 'UNK')
            res2_name = self.residue_info.get(c[1], 'UNK')
            print(f"{res1_name}{c[0][1]}-{res2_name}{c[1][1]}")
        
        if len(contacts_list) > 10:
            print(f"... and {len(contacts_list) - 10} more contacts (see text file for complete list)")

        return contact_matrix, analysis_info
    
    def _write_contact_interactions(self, contacts_list: List[Tuple], pdb_filename: str):
        """Write detailed contact interactions to text file"""
        base_name = Path(pdb_filename).stem
        cutoff_str = f"{self.config.cutoff:.1f}A".replace('.', 'p')
        
        # Single detailed interactions file
        detailed_file = f"{base_name}_interactions_{cutoff_str}.txt"
        with open(detailed_file, "w") as f:
            f.write("DETAILED CONTACT INTERACTIONS\n")
            f.write("=" * 50 + "\n")
            f.write(f"PDB File: {pdb_filename}\n")
            f.write(f"Distance Cutoff: {self.config.cutoff} Å\n")
            f.write(f"Total Contacts: {len(contacts_list)}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("FORMAT: Residue1-Number ↔ Residue2-Number [Contact Type]\n\n")
            
            # Sort contacts for better readability
            sorted_contacts = sorted(contacts_list, key=lambda x: (x[0][1], x[1][1]))
            
            intra_count = 0
            inter_count = 0
            
            for contact in sorted_contacts:
                key1, key2 = contact
                chain1, res1 = key1
                chain2, res2 = key2
                
                res1_name = self.residue_info.get(key1, 'UNK')
                res2_name = self.residue_info.get(key2, 'UNK')
                
                contact_type = "INTRA-CHAIN" if chain1 == chain2 else "INTER-CHAIN"
                if contact_type == "INTRA-CHAIN":
                    intra_count += 1
                else:
                    inter_count += 1
                
                f.write(f"{res1_name}{res1}-{res2_name}{res2} [{contact_type}]\n")
            
            f.write(f"\nSUMMARY:\n")
            f.write(f"Intra-chain contacts: {intra_count}\n")
            f.write(f"Inter-chain contacts: {inter_count}\n")
        
        self.logger.info(f"Contact interactions written to: {detailed_file}")

class QuadrantContactVisualizer:
    """Specialized visualization for quadrant contact map"""
    
    def __init__(self, config: FibrilConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.setup_style()
    
    def setup_style(self):
        """Setup professional visualization style"""
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 9,
            'figure.titlesize': 14,
            'figure.dpi': 100,
            'savefig.dpi': 300
        })
    
    def plot_quadrant_contact_map(self, contact_matrix: np.ndarray, analysis_info: Dict) -> bool:
        """Create quadrant contact map showing inter and intra-chain contacts"""
        try:
            with timer("Quadrant contact map visualization", self.logger):
                chains = analysis_info['chains']
                chain_boundaries = analysis_info['chain_boundaries']
                global_residue_order = analysis_info['global_residue_order']
                matrix_size = analysis_info['matrix_size']
                
                if len(chains) < 2:
                    self.logger.warning("Need at least 2 chains for quadrant visualization")
                    return self._plot_single_chain_fallback(contact_matrix, analysis_info)
                
                # Create figure
                fig, ax = plt.subplots(1, 1, figsize=self.config.figsize_heatmap)
                
                display_matrix = np.zeros_like(contact_matrix, dtype=float)
                for i in range(matrix_size):
                    for j in range(matrix_size):
                        if contact_matrix[i, j] > 0:
                            chain_i, res_i = analysis_info['global_residue_order'][i]
                            chain_j, res_j = analysis_info['global_residue_order'][j]
                            display_matrix[i, j] = 1.0 if chain_i == chain_j else 2.0

                # Create custom colormap: white (0), blue (1), red (2)
                from matplotlib.colors import ListedColormap, BoundaryNorm
                
                colors = ['white', 'blue', 'red']
                n_bins = 3
                cmap = ListedColormap(colors[:n_bins])
                bounds = [0, 0.5, 1.5, 2.5]
                norm = BoundaryNorm(bounds, cmap.N)
                
                # Plot the matrix
                im = ax.imshow(display_matrix, cmap=cmap, norm=norm, aspect='equal', origin='lower')
                
                # Add chain boundary lines
                for chain in chains:
                    start, end = chain_boundaries[chain]
                    if start > 0:
                        ax.axhline(y=start-0.5, color='black', linewidth=2, alpha=0.8)
                        ax.axvline(x=start-0.5, color='black', linewidth=2, alpha=0.8)
                
                # Configure ticks and labels
                self._configure_ticks_and_labels(ax, analysis_info)
                
                # Add title and labels
                ax.set_title(f'Inter vs Intra-Chain Contact Map\nCutoff: {self.config.cutoff} Å', 
                           fontsize=14, pad=20)
                ax.set_xlabel('Residue Number', fontsize=12)
                ax.set_ylabel('Residue Number', fontsize=12)
                
                # Add legend
                self._add_contact_legend(ax)
                
                # Add chain labels
                self._add_chain_labels(ax, analysis_info)
                
                # Add statistics text
                self._add_statistics_text(ax, display_matrix, analysis_info)
                
                plt.tight_layout()
                
                # Save figure
                base_name = Path(self.config.pdb_filename).stem
                cutoff_str = f"{self.config.cutoff:.1f}A".replace('.', 'p')
                output_file = f"{base_name}_quadrant_contact_map_{cutoff_str}.png"
                plt.savefig(output_file, dpi=self.config.dpi, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                self.logger.info(f"Quadrant contact map saved: {output_file}")
                
                plt.show()
                return True
                
        except Exception as e:
            self.logger.error(f"Error generating quadrant contact map: {e}")
            return False
    
    def _configure_ticks_and_labels(self, ax, analysis_info: Dict):
        """Configure axis ticks and labels"""
        chains = analysis_info['chains']
        chain_boundaries = analysis_info['chain_boundaries']
        global_residue_order = analysis_info['global_residue_order']
        matrix_size = analysis_info['matrix_size']

        tick_positions = []
        tick_labels = []

        for chain in chains:
            start, end = chain_boundaries[chain]
            if start < matrix_size:
                tick_positions.append(start)
                tick_labels.append(f"{global_residue_order[start][1]}")

            chain_length = end - start
            if chain_length > 20:
                mid = start + chain_length // 2
                if mid < matrix_size:
                    tick_positions.append(mid)
                    tick_labels.append(f"{global_residue_order[mid][1]}")
        
        # Limit number of ticks to avoid crowding
        if len(tick_positions) > 15:
            step = len(tick_positions) // 10
            tick_positions = tick_positions[::step]
            tick_labels = tick_labels[::step]
        
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels)
    
    def _add_contact_legend(self, ax):
        """Add legend for contact types"""
        from matplotlib.patches import Rectangle
        
        intra_patch = Rectangle((0, 0), 1, 1, facecolor='blue', alpha=0.8, label='Intra-chain contacts')
        inter_patch = Rectangle((0, 0), 1, 1, facecolor='red', alpha=0.8, label='Inter-chain contacts')
        
        ax.legend(handles=[intra_patch, inter_patch], loc='upper right', 
                 bbox_to_anchor=(0.98, 0.98), frameon=True, fancybox=True, shadow=True)
    
    def _add_chain_labels(self, ax, analysis_info: Dict):
        """Add chain labels to the plot"""
        chains = analysis_info['chains']
        chain_boundaries = analysis_info['chain_boundaries']
        matrix_size = analysis_info['matrix_size']
        
        for chain in chains:
            start, end = chain_boundaries[chain]
            center = (start + end) / 2
            
            if center < matrix_size:
                ax.text(center, center, f'Chain {chain}', 
                       ha='center', va='center', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    def _add_statistics_text(self, ax, display_matrix: np.ndarray, analysis_info: Dict):
        """Add contact statistics to the plot"""
        total_contacts = np.count_nonzero(display_matrix)
        intra_contacts = np.count_nonzero(display_matrix == 1.0)
        inter_contacts = np.count_nonzero(display_matrix == 2.0)
        
        stats_text = f"Total contacts: {total_contacts}\n"
        stats_text += f"Intra-chain: {intra_contacts} ({intra_contacts/max(1,total_contacts)*100:.1f}%)\n"
        stats_text += f"Inter-chain: {inter_contacts} ({inter_contacts/max(1,total_contacts)*100:.1f}%)"
        
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, 
               fontsize=9, verticalalignment='bottom',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))
    
    def _plot_single_chain_fallback(self, contact_matrix: np.ndarray, analysis_info: Dict) -> bool:
        """Fallback visualization for single chain structures"""
        try:
            fig, ax = plt.subplots(1, 1, figsize=self.config.figsize_heatmap)
            
            im = ax.imshow(contact_matrix, cmap='Blues', aspect='equal', origin='lower')
            
            ax.set_title(f'Contact Map (Single Chain)\nCutoff: {self.config.cutoff} Å', 
                        fontsize=14, pad=20)
            ax.set_xlabel('Residue Number', fontsize=12)
            ax.set_ylabel('Residue Number', fontsize=12)
            
            plt.colorbar(im, ax=ax, shrink=0.8)
            plt.tight_layout()
            
            base_name = Path(self.config.pdb_filename).stem
            cutoff_str = f"{self.config.cutoff:.1f}A".replace('.', 'p')
            output_file = f"{base_name}_single_chain_contact_map_{cutoff_str}.png"
            plt.savefig(output_file, dpi=self.config.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            
            plt.show()
            return True
            
        except Exception as e:
            self.logger.error(f"Error in single chain fallback: {e}")
            return False

class QuadrantContactAnalyzer:
    """Main analyzer class for quadrant contact visualization"""
    
    def __init__(self, config: FibrilConfig):
        self.config = config
        self.logger = setup_logging()
        self.processor = FibrilPDBProcessor(config, self.logger)
        self.visualizer = QuadrantContactVisualizer(config, self.logger)
    
    def run_quadrant_analysis(self) -> bool:
        """Run contact analysis with quadrant visualization"""
        self.logger.info("Starting Quadrant Contact Map Analysis")
        
        if not self.config.validate():
            self.logger.error("Configuration validation failed")
            return False
        
        try:
            # Load and analyze structure
            contact_matrix, analysis_info = self.processor.analyze_structure_with_chain_separation(
                self.config.pdb_filename)
            
            if contact_matrix.size == 0:
                self.logger.error("Failed to load or process structure")
                return False
            
            # Generate quadrant visualization
            success = self.visualizer.plot_quadrant_contact_map(contact_matrix, analysis_info)
            
            if success:
                self.logger.info("Quadrant contact analysis completed successfully!")
                self.print_summary(contact_matrix, analysis_info)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Critical error in quadrant analysis: {e}")
            return False
    
    def print_summary(self, contact_matrix: np.ndarray, analysis_info: Dict):
        """Print analysis summary"""
        chains = analysis_info['chains']
        total_contacts = np.count_nonzero(contact_matrix)
        contacts_list = analysis_info.get('contacts_list', [])
        
        # Count inter vs intra contacts
        intra_contacts = 0
        inter_contacts = 0
        
        for contact in contacts_list:
            key1, key2 = contact
            if key1[0] == key2[0]:
                intra_contacts += 1
            else:
                inter_contacts += 1
        
        print("\n" + "="*60)
        print("QUADRANT CONTACT MAP SUMMARY")
        print("="*60)
        print(f"Structure: {self.config.pdb_filename}")
        print(f"Number of chains: {len(chains)}")
        print(f"Chain IDs: {', '.join(chains)}")
        print(f"Distance cutoff: {self.config.cutoff} Å")
        print(f"Total contacts: {total_contacts:,}")
        print(f"  - Intra-chain: {intra_contacts:,} ({intra_contacts/max(1,total_contacts)*100:.1f}%)")
        print(f"  - Inter-chain: {inter_contacts:,} ({inter_contacts/max(1,total_contacts)*100:.1f}%)")
        print("\nOutput files generated:")
        
        base_name = Path(self.config.pdb_filename).stem
        cutoff_str = f"{self.config.cutoff:.1f}A".replace('.', 'p')
        
        print(f"  - Visualization: {base_name}_quadrant_contact_map_{cutoff_str}.png")
        print(f"  - Interactions: {base_name}_interactions_{cutoff_str}.txt")
        print("="*60)

# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================

def analyze_quadrant_contacts(pdb_file: str = "6WQK.pdb", cutoff: float = 5.0) -> bool:
    """Main function to run quadrant contact analysis"""
    print(f"Running Simplified Quadrant Contact Analysis")
    print(f"PDB file: {pdb_file}")
    print(f"Distance cutoff: {cutoff} Å")
    print("="*50)
    
    if not Path(pdb_file).exists():
        print(f"Error: {pdb_file} not found!")
        return False
    
    # Configure analysis
    config = FibrilConfig()
    config.pdb_filename = pdb_file
    config.cutoff = cutoff
    
    # Run analysis
    analyzer = QuadrantContactAnalyzer(config)
    return analyzer.run_quadrant_analysis()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    try:
        # Check dependencies
        print("Checking dependencies...")
        if MDA_AVAILABLE:
            print("MDAnalysis available")
        else:
            print("MDAnalysis not available - using fallback parser")
        
        print("Required packages: pandas, matplotlib, seaborn, numpy")
        print()
        
        # Run quadrant analysis
        success = analyze_quadrant_contacts(cutoff=5.0)
        
        if success:
            print("\nAnalysis completed successfully!")
            print("Check the generated files:")
            print("  - PNG: Visualization of the contact map")
            print("  - TXT: Detailed interaction list")
        else:
            print("\nAnalysis failed. Check console output for details.")
    
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
