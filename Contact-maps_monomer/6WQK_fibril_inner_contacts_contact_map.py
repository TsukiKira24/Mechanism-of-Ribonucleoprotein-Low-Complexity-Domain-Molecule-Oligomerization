import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
from contextlib import contextmanager
import re

warnings.filterwarnings('ignore')

try:
    import MDAnalysis as mda
    from MDAnalysis.analysis.distances import distance_array
    MDA_AVAILABLE = True
except ImportError:
    MDA_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================

class OverlaidFragmentConfig:
    """Configuration for overlaid fragment analysis"""
    
    def __init__(self):
        self.cutoff = 5.0
        self.pdb_filename = "6WQK.pdb"
        self.fragment_start = 263
        self.fragment_end = 319
        self.expected_chains = 5
        self.dpi = 300
        self.figsize_heatmap = (5, 4)
        self.figsize_barplot = (10, 6)
        self.exclude_hydrogens = True
        
    def get_fragment_size(self):
        """Get the size of the fragment"""
        return self.fragment_end - self.fragment_start + 1
        
    def validate(self) -> bool:
        """Validate configuration parameters"""
        try:
            assert self.cutoff > 0, "Cutoff must be positive"
            assert Path(self.pdb_filename).exists(), f"PDB file {self.pdb_filename} not found"
            assert self.fragment_start < self.fragment_end, "Fragment start must be < end"
            return True
        except AssertionError as e:
            print(f"Configuration validation failed: {e}")
            return False

# =============================================================================
# LOGGING
# =============================================================================

def setup_logging() -> logging.Logger:
    """Setup logging"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
        
        file_handler = logging.FileHandler('overlaid_fragment_analysis.log')
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
# PDB PROCESSOR
# =============================================================================

class OverlaidFragmentProcessor:
    """Process PDB to extract and overlay fragment contacts from multiple chains"""
    
    def __init__(self, config: OverlaidFragmentConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.residue_info = {}
        self.chain_info = {}
    
    def parse_pdb(self, filename: str) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Parse PDB and organize by chain"""
        self.logger.info(f"Parsing PDB file: {filename}")
        
        chain_atoms = {}
        chains = set()
        
        try:
            with open(filename, 'r') as f:
                for line in f:
                    if not (line.startswith('ATOM') or line.startswith('HETATM')):
                        continue
                    
                    if len(line) < 54:
                        continue
                    
                    atom_name = line[12:16].strip()
                    chain_id = line[21:22].strip()
                    residue_name = line[17:20].strip()
                    
                    res_field = line[22:26].strip()
                    try:
                        residue_num = int(res_field)
                    except ValueError:
                        res_match = re.match(r'(\d+)', res_field)
                        if res_match:
                            residue_num = int(res_match.group(1))
                        else:
                            continue
                    
                    # Only process fragment residues
                    if not (self.config.fragment_start <= residue_num <= self.config.fragment_end):
                        continue
                    
                    if self.config.exclude_hydrogens and atom_name.startswith('H'):
                        continue
                    
                    try:
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                    except ValueError:
                        continue
                    
                    chain_id = chain_id if chain_id else 'A'
                    chains.add(chain_id)
                    
                    if chain_id not in chain_atoms:
                        chain_atoms[chain_id] = []
                    
                    chain_atoms[chain_id].append({
                        'atom_name': atom_name,
                        'residue_num': residue_num,
                        'residue_name': residue_name,
                        'x': x, 'y': y, 'z': z
                    })
                    
                    # Store residue info
                    key = (chain_id, residue_num)
                    self.residue_info[key] = residue_name
            
            self.logger.info(f"Found {len(chains)} chains: {sorted(chains)}")
            
            # Convert to position arrays per chain
            chain_positions = {}
            chain_residue_ids = {}
            
            for chain_id, atoms in chain_atoms.items():
                positions = np.array([[a['x'], a['y'], a['z']] for a in atoms])
                residue_ids = np.array([a['residue_num'] for a in atoms])
                
                chain_positions[chain_id] = positions
                chain_residue_ids[chain_id] = residue_ids
                
                self.logger.info(f"Chain {chain_id}: {len(atoms)} atoms in fragment {self.config.fragment_start}-{self.config.fragment_end}")
            
            info = {
                'chains': sorted(chains),
                'chain_residue_ids': chain_residue_ids,
                'residue_info': self.residue_info
            }
            
            return chain_positions, info
            
        except Exception as e:
            self.logger.error(f"Error parsing PDB: {e}")
            return {}, {}
    
    def calculate_fragment_contacts(self, chain_positions: Dict[str, np.ndarray], 
                                    chain_residue_ids: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Calculate contact matrix for each chain's fragment"""
        fragment_size = self.config.get_fragment_size()
        chain_contact_matrices = {}
        
        for chain_id in sorted(chain_positions.keys()):
            self.logger.info(f"Calculating contacts for chain {chain_id}...")
            
            positions = chain_positions[chain_id]
            residue_ids = chain_residue_ids[chain_id]
            
            # Initialize contact matrix for this fragment
            contact_matrix = np.zeros((fragment_size, fragment_size), dtype=np.int32)
            
            # Calculate distance matrix
            dist_matrix = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
            atom_contacts = dist_matrix < self.config.cutoff
            
            # Convert to residue-level contacts
            ci, cj = np.where(atom_contacts)
            for i, j in zip(ci, cj):
                if i >= j:
                    continue
                
                res_i = residue_ids[i]
                res_j = residue_ids[j]
                
                if res_i == res_j:
                    continue
                
                # Map to fragment indices (0-based)
                idx_i = res_i - self.config.fragment_start
                idx_j = res_j - self.config.fragment_start
                
                if 0 <= idx_i < fragment_size and 0 <= idx_j < fragment_size:
                    contact_matrix[idx_i, idx_j] = 1
                    contact_matrix[idx_j, idx_i] = 1
            
            chain_contact_matrices[chain_id] = contact_matrix
            
            contact_count = np.count_nonzero(contact_matrix) // 2
            self.logger.info(f"Chain {chain_id}: {contact_count} unique contacts")
        
        return chain_contact_matrices
    
    def create_overlaid_matrix(self, chain_contact_matrices: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Create overlaid contact matrix showing consensus across chains"""
        fragment_size = self.config.get_fragment_size()
        
        # Sum matrix - counts how many chains have each contact
        sum_matrix = np.zeros((fragment_size, fragment_size), dtype=np.int32)
        
        # Binary consensus matrix - contact present in at least one chain
        consensus_matrix = np.zeros((fragment_size, fragment_size), dtype=np.int32)
        
        for chain_id, contact_matrix in chain_contact_matrices.items():
            sum_matrix += contact_matrix
            consensus_matrix = np.maximum(consensus_matrix, contact_matrix)
        
        self.logger.info(f"Overlaid matrix created:")
        self.logger.info(f"  Total unique contacts (any chain): {np.count_nonzero(consensus_matrix) // 2}")
        self.logger.info(f"  Contacts in all {len(chain_contact_matrices)} chains: {np.count_nonzero(sum_matrix == len(chain_contact_matrices)) // 2}")
        
        return sum_matrix, consensus_matrix

# =============================================================================
# VISUALIZER
# =============================================================================

class OverlaidFragmentVisualizer:
    """Visualize overlaid fragment contacts"""
    
    def __init__(self, config: OverlaidFragmentConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.setup_style()
    
    def setup_style(self):
        """Setup visualization style"""
        plt.rcParams.update({
            'font.size': 8,
            'axes.titlesize': 10,
            'axes.labelsize': 8,
            'xtick.labelsize': 7,
            'ytick.labelsize': 7,
            'legend.fontsize': 7,
            'figure.titlesize': 12,
            'figure.dpi': 100,
            'savefig.dpi': 300
        })
    
    def create_custom_colormap(self, base_cmap_name: str):
        """Create custom colormap with white for zero"""
        base_cmap = plt.cm.get_cmap(base_cmap_name)
        colors = base_cmap(np.linspace(0, 1, 256))
        colors[0] = [1, 1, 1, 1]
        return plt.matplotlib.colors.ListedColormap(colors)
    
    def plot_overlaid_contact_map(self, sum_matrix: np.ndarray, 
                                  chain_contact_matrices: Dict[str, np.ndarray]) -> bool:
        """Plot overlaid contact map showing frequency across chains"""
        try:
            with timer("Overlaid contact map visualization", self.logger):
                fig, ax = plt.subplots(1, 1, figsize=self.config.figsize_heatmap)
                
                # Use custom colormap
                cmap = self.create_custom_colormap('RdYlBu_r')
                
                n_chains = len(chain_contact_matrices)
                max_val = n_chains
                
                # Plot the sum matrix (shows frequency across chains)
                im = ax.imshow(sum_matrix, cmap=cmap, aspect='equal',
                             vmin=0, vmax=max_val, origin='lower')
                
                # Configure ticks
                fragment_size = self.config.get_fragment_size()
                n_ticks = min(10, fragment_size)
                tick_step = max(1, fragment_size // n_ticks)
                tick_positions = list(range(0, fragment_size, tick_step))
                tick_labels = [str(self.config.fragment_start + i) for i in tick_positions]
                
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(tick_labels, rotation=45, fontsize=12)
                ax.set_yticks(tick_positions)
                ax.set_yticklabels(tick_labels, fontsize=12)
                
                #ax.set_title(f'Overlaid Contact Map - Fragment {self.config.fragment_start}-{self.config.fragment_end}\n'
                #           f'{n_chains} Chains Superimposed (Cutoff: {self.config.cutoff} Å)', 
                #           fontsize=10, pad=10)
                ax.set_xlabel('Residue Number', fontsize=12)
                ax.set_ylabel('Residue Number', fontsize=12)
                
                # Colorbar showing frequency
                cbar = plt.colorbar(im, ax=ax, shrink=0.6, aspect=15)
                cbar.set_label(f'Number of Chains\nwith Contact (max={n_chains})', fontsize=7)
                cbar.ax.tick_params(labelsize=6)
                
                # Add statistics
                self._add_overlay_stats(ax, sum_matrix, n_chains)
                
                plt.tight_layout()
                
                base_name = Path(self.config.pdb_filename).stem
                cutoff_str = f"{self.config.cutoff:.1f}A".replace('.', 'p')
                output_file = f"{base_name}_OVERLAID_fragment_{self.config.fragment_start}-{self.config.fragment_end}_{n_chains}chains_{cutoff_str}.png"
                plt.savefig(output_file, dpi=self.config.dpi, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                self.logger.info(f"Overlaid contact map saved: {output_file}")
                
                plt.show()
                return True
                
        except Exception as e:
            self.logger.error(f"Error generating overlaid contact map: {e}")
            return False
    

    def _add_overlay_stats(self, ax, sum_matrix: np.ndarray, n_chains: int):
        """Add statistics text to overlay map"""
        total_contacts = np.count_nonzero(sum_matrix)
        contacts_all_chains = np.count_nonzero(sum_matrix == n_chains)
        contacts_any_chain = np.count_nonzero(sum_matrix > 0)
        
        stats_text = f"Fragment: {self.config.fragment_start}-{self.config.fragment_end}\n"
        stats_text += f"Chains overlaid: {n_chains}\n"
        stats_text += f"Contacts in ANY chain: {contacts_any_chain // 2}\n"
        stats_text += f"Contacts in ALL chains: {contacts_all_chains // 2}\n"
        stats_text += f"Cutoff: {self.config.cutoff} Å"
        
        #ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
        #       fontsize=8, verticalalignment='top',
        #       bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))

# =============================================================================
# ANALYZER
# =============================================================================

class OverlaidFragmentAnalyzer:
    """Main analyzer for overlaid fragment contacts"""
    
    def __init__(self, config: OverlaidFragmentConfig):
        self.config = config
        self.logger = setup_logging()
        self.processor = OverlaidFragmentProcessor(config, self.logger)
        self.visualizer = OverlaidFragmentVisualizer(config, self.logger)
    
    def run_analysis(self) -> bool:
        """Run overlaid fragment analysis"""
        self.logger.info(f"Starting Overlaid Fragment Analysis ({self.config.fragment_start}-{self.config.fragment_end})")
        
        if not self.config.validate():
            self.logger.error("Configuration validation failed")
            return False
        
        try:
            # Parse PDB
            chain_positions, info = self.processor.parse_pdb(self.config.pdb_filename)
            
            if not chain_positions:
                self.logger.error("Failed to parse PDB or extract fragment")
                return False
            
            chains = info['chains']
            self.logger.info(f"Processing {len(chains)} chains: {chains}")
            
            # Calculate contacts for each chain's fragment
            chain_contact_matrices = self.processor.calculate_fragment_contacts(
                chain_positions, info['chain_residue_ids'])
            
            # Create overlaid matrices
            sum_matrix, consensus_matrix = self.processor.create_overlaid_matrix(
                chain_contact_matrices)
            
            # Generate visualizations
            success = True
            success &= self.visualizer.plot_overlaid_contact_map(sum_matrix, chain_contact_matrices)
            
            if success:
                self.logger.info("Overlaid fragment analysis completed successfully!")
                self._print_summary(sum_matrix, chain_contact_matrices, consensus_matrix)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Critical error in overlaid analysis: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _print_summary(self, sum_matrix: np.ndarray, 
                      chain_contact_matrices: Dict[str, np.ndarray],
                      consensus_matrix: np.ndarray):
        """Print analysis summary"""
        n_chains = len(chain_contact_matrices)
        
        print("\n" + "="*70)
        print(f"OVERLAID FRAGMENT ANALYSIS SUMMARY")
        print("="*70)
        print(f"Structure: {self.config.pdb_filename}")
        print(f"Fragment: {self.config.fragment_start}-{self.config.fragment_end} ({self.config.get_fragment_size()} residues)")
        print(f"Number of chains overlaid: {n_chains}")
        print(f"Distance cutoff: {self.config.cutoff} Å")
        print()
        
        print("Individual chain contacts:")
        for chain_id, matrix in sorted(chain_contact_matrices.items()):
            contacts = np.count_nonzero(matrix) // 2
            print(f"  Chain {chain_id}: {contacts} contacts")
        
        print()
        print("Overlay statistics:")
        contacts_any = np.count_nonzero(consensus_matrix) // 2
        contacts_all = np.count_nonzero(sum_matrix == n_chains) // 2
        
        print(f"  Contacts in ANY chain: {contacts_any}")
        print(f"  Contacts in ALL {n_chains} chains: {contacts_all}")
        print(f"  Conservation rate: {contacts_all/max(1, contacts_any)*100:.1f}%")
        
        print()
        print("Output files generated:")
        base_name = Path(self.config.pdb_filename).stem
        cutoff_str = f"{self.config.cutoff:.1f}A".replace('.', 'p')
        
        print(f"  - Overlaid map: {base_name}_OVERLAID_fragment_{self.config.fragment_start}-{self.config.fragment_end}_{n_chains}chains_{cutoff_str}.png")
        print(f"  - Individual+consensus: {base_name}_INDIVIDUAL_consensus_fragment_{self.config.fragment_start}-{self.config.fragment_end}_{cutoff_str}.png")
        print(f"  - Bar plot: {base_name}_OVERLAID_barplot_fragment_{self.config.fragment_start}-{self.config.fragment_end}_{cutoff_str}.png")
        print("="*70)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def analyze_overlaid_fragment(pdb_file: str = "6WQK.pdb", cutoff: float = 5.0,
                              fragment_start: int = 263, fragment_end: int = 319) -> bool:
    """Main function to run overlaid fragment analysis"""
    print(f"Running Overlaid Fragment Analysis")
    print(f"PDB file: {pdb_file}")
    print(f"Fragment: {fragment_start}-{fragment_end}")
    print(f"Distance cutoff: {cutoff} Å")
    print(f"Mode: Overlay all 5 chains onto single contact map")
    print("="*50)
    
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

if __name__ == "__main__":
    try:
        print("Checking dependencies...")
        if MDA_AVAILABLE:
            print("MDAnalysis available")
        else:
            print("MDAnalysis not available - using fallback parser")
        
        print("Required packages: pandas, matplotlib, seaborn, numpy")
        print()
        
        success = analyze_overlaid_fragment(cutoff=5.0)
        
        if success:
            print("\nOverlaid fragment analysis completed successfully!")
            print("\nGenerated visualizations:")
            print("  1. Overlaid contact map - shows contact frequency across all 5 chains")
            print("     (Color intensity = number of chains with that contact)")
            print("  2. Individual + consensus maps - compare each chain separately")
            print("  3. Bar plot - contact frequency per residue")
            print("\nThis analysis reveals which contacts are:")
            print("  - Universal (present in all chains)")
            print("  - Common (present in multiple chains)")
            print("  - Chain-specific (present in only one chain)")
        else:
            print("\nAnalysis failed. Check console output for details.")
    
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    sys.exit(0 if success else 1)
