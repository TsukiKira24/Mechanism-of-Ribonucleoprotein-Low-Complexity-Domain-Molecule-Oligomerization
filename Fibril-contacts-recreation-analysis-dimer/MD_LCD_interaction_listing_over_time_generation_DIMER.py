# -*- coding: utf-8 -*-
"""
Dimer Protein Interaction Analysis - Time Series Analysis
Analyzes contacts for two-protein system (Protein A and Protein B)
Tracks A-A, B-B, and A-B interactions over time
"""

import pandas as pd
import numpy as np
import os
import re
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array
from collections import defaultdict

# === CONFIGURATION ===
cutoff = 5.0  # distance cutoff in Å
FOLD = '2'

# Dimer configuration - two chains
chain_size = 154  # Size of each chain
start_residue = 188  # Starting residue number for display
end_residue = 341    # Ending residue number for display

# Chain definitions
CHAIN_A_START = 0      # Chain A starts at index 0
CHAIN_A_END = 153      # Chain A ends at index 153 (154 residues)
CHAIN_B_START = 154    # Chain B starts at index 154
CHAIN_B_END = 307      # Chain B ends at index 307 (154 residues)

pH_conditions = [
    {'PH': '40', 'PHdot': '4.0'},
    {'PH': '74', 'PHdot': '7.4'},
    {'PH': '85', 'PHdot': '8.5'}
]

def index_to_residue_info(idx, chain='A'):
    """Convert matrix index to residue information"""
    if chain == 'A':
        if idx < CHAIN_A_START or idx > CHAIN_A_END:
            return None, None, None
        local_idx = idx - CHAIN_A_START
    else:  # chain == 'B'
        if idx < CHAIN_B_START or idx > CHAIN_B_END:
            return None, None, None
        local_idx = idx - CHAIN_B_START
    
    data_res_num = local_idx + 1
    display_res_num = start_residue + local_idx
    
    return data_res_num, display_res_num, chain

def get_chain_from_index(idx):
    """Determine which chain an index belongs to"""
    if CHAIN_A_START <= idx <= CHAIN_A_END:
        return 'A'
    elif CHAIN_B_START <= idx <= CHAIN_B_END:
        return 'B'
    else:
        return None

def extract_pdb_timestamps(pdb_filename):
    """Extract timestamps from PDB file TITLE records"""
    timestamps = []
    
    if not os.path.exists(pdb_filename):
        print(f" ERROR: PDB file '{pdb_filename}' not found.")
        return timestamps
    
    with open(pdb_filename, 'r') as f:
        for line in f:
            if line.startswith('TITLE'):
                match = re.search(r't=\s*(\d+\.?\d*)', line)
                if match:
                    time_ps = float(match.group(1))
                    time_ns = time_ps / 1000.0
                    timestamps.append(time_ns)
    
    return timestamps

def get_residue_name_from_pdb(pdb_file, chain, residue_num):
    """Extract residue name from PDB file for specific chain"""
    try:
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith('ATOM'):
                    line_chain = line[21:22].strip()
                    line_resnum = int(line[22:26].strip())
                    
                    # Match chain and residue number
                    if line_chain == chain and line_resnum == residue_num:
                        return line[17:20].strip()
    except:
        pass
    return "UNK"

def create_residue_mapping(pdb_file):
    """Create mapping of indices to residue names for both chains"""
    residue_mapping = {}
    
    # Map Chain A residues
    for i in range(CHAIN_A_START, CHAIN_A_END + 1):
        local_idx = i - CHAIN_A_START
        data_res_num = local_idx + 1
        res_name = get_residue_name_from_pdb(pdb_file, 'A', data_res_num)
        residue_mapping[i] = res_name
    
    # Map Chain B residues
    for i in range(CHAIN_B_START, CHAIN_B_END + 1):
        local_idx = i - CHAIN_B_START
        data_res_num = local_idx + 1
        res_name = get_residue_name_from_pdb(pdb_file, 'B', data_res_num)
        residue_mapping[i] = res_name
    
    return residue_mapping

def get_residue_display_name(idx, residue_mapping):
    """Get display name for residue with chain prefix"""
    chain = get_chain_from_index(idx)
    if chain is None:
        return None
    
    data_res_num, display_res_num, _ = index_to_residue_info(idx, chain)
    if data_res_num is None:
        return None
    
    res_name = residue_mapping.get(idx, 'UNK')
    return f"{chain}_{res_name}{display_res_num}"

def classify_interaction_type(chain_i, chain_j):
    """Classify interaction as A-A, B-B, or A-B"""
    if chain_i == 'A' and chain_j == 'A':
        return 'A_internal'
    elif chain_i == 'B' and chain_j == 'B':
        return 'B_internal'
    elif (chain_i == 'A' and chain_j == 'B') or (chain_i == 'B' and chain_j == 'A'):
        return 'A_B_inter'
    else:
        return 'unknown'

def process_pdb_contacts_per_frame(pdb_filename, residue_mapping):
    """Process PDB file and return contacts for each frame, classified by type"""
    if not os.path.exists(pdb_filename):
        print(f" ERROR: PDB file '{pdb_filename}' not found.")
        return {}

    timestamps = extract_pdb_timestamps(pdb_filename)
    
    u = mda.Universe(pdb_filename, format='PDB')
    n_frames = len(u.trajectory)
    
    print(f"   Found {n_frames} frames in PDB file")
    print(f"   Extracted {len(timestamps)} timestamps from headers")
    
    if len(timestamps) < n_frames:
        print(f"   Warning: Only {len(timestamps)} timestamps found for {n_frames} frames")
        print(f"   Using frame index + 1 as nanosecond number")
        timestamps = [i + 1 for i in range(n_frames)]
    
    frame_contacts = {}
    
    for frame_idx, ts in enumerate(u.trajectory):
        if frame_idx % 100 == 0:
            print(f"   Processing frame {frame_idx + 1}/{n_frames}")
        
        if frame_idx < len(timestamps):
            frame_time = timestamps[frame_idx]
        else:
            frame_time = frame_idx + 1
        
        try:
            protein_atoms = u.select_atoms("protein and not name H*")
        except:
            protein_atoms = u.select_atoms("not name H*")
        
        if len(protein_atoms) == 0:
            frame_contacts[frame_time] = {
                'A_internal': [],
                'B_internal': [],
                'A_B_inter': []
            }
            continue

        # Get residue IDs and positions
        residue_ids = protein_atoms.resids - 1
        chain_ids = [c for c in protein_atoms.chainIDs]
        positions = protein_atoms.positions
        
        # Convert chain letters to indices
        residue_indices = []
        for res_id, chain_id in zip(residue_ids, chain_ids):
            if chain_id == 'A':
                idx = CHAIN_A_START + res_id
            elif chain_id == 'B':
                idx = CHAIN_B_START + res_id
            else:
                idx = -1  # Invalid
            residue_indices.append(idx)
        
        residue_indices = np.array(residue_indices)
        
        # Filter valid indices
        valid_mask = (residue_indices >= 0) & (residue_indices <= CHAIN_B_END)
        if not np.any(valid_mask):
            frame_contacts[frame_time] = {
                'A_internal': [],
                'B_internal': [],
                'A_B_inter': []
            }
            continue
        
        residue_indices = residue_indices[valid_mask]
        positions = positions[valid_mask]
        
        # Calculate distance matrix
        dist_matrix = distance_array(positions, positions)
        contact_matrix = dist_matrix < cutoff

        # Find contacts
        atom_contacts = np.where(contact_matrix)
        contacted_pairs = set()
        
        for i, j in zip(atom_contacts[0], atom_contacts[1]):
            if i != j:
                res_i = residue_indices[i]
                res_j = residue_indices[j]
                if res_i != res_j:  # Exclude self-interactions
                    pair = tuple(sorted([res_i, res_j]))
                    contacted_pairs.add(pair)
        
        # Classify contacts by type
        classified_contacts = {
            'A_internal': [],
            'B_internal': [],
            'A_B_inter': []
        }
        
        for res_i, res_j in contacted_pairs:
            chain_i = get_chain_from_index(res_i)
            chain_j = get_chain_from_index(res_j)
            
            if chain_i is None or chain_j is None:
                continue
            
            interaction_type = classify_interaction_type(chain_i, chain_j)
            
            res_name_i = get_residue_display_name(res_i, residue_mapping)
            res_name_j = get_residue_display_name(res_j, residue_mapping)
            
            if res_name_i and res_name_j:
                contact = f"{res_name_i}-{res_name_j}"
                classified_contacts[interaction_type].append(contact)
        
        # Sort contacts alphabetically within each type
        for contact_type in classified_contacts:
            classified_contacts[contact_type] = sorted(classified_contacts[contact_type])
        
        frame_contacts[frame_time] = classified_contacts
    
    return frame_contacts

def save_contact_files(frame_contacts, FOLD, PH, PHdot):
    """Save contact data to separate files for each interaction type"""
    
    print(f"\n Saving contact files for pH {PHdot}...")
    files_written = []
    
    contact_types = {
        'A_internal': 'ProtA_internal',
        'B_internal': 'ProtB_internal',
        'A_B_inter': 'ProtA_ProtB_inter'
    }
    
    for contact_type, file_label in contact_types.items():
        filename = f"FOLD{FOLD}_pH{PH}_{file_label}_interactions.txt"
        
        with open(filename, 'w') as f:
            # Write header
            f.write(f"Protein Interaction Analysis - {file_label.replace('_', ' ').title()} - pH {PHdot}\n")
            f.write("="*70 + "\n")
            f.write(f"Distance cutoff: {cutoff} Å\n")
            f.write(f"Format: Each time point shows all contacts of type {contact_type}\n")
            f.write("-"*70 + "\n\n")
            
            # Write data for each frame
            for frame_time in sorted(frame_contacts.keys()):
                contacts = frame_contacts[frame_time][contact_type]
                
                # Write time header
                if frame_time == int(frame_time):
                    f.write(f"t= {int(frame_time)} ns\n")
                else:
                    f.write(f"t= {frame_time:.1f} ns\n")
                
                # Write contacts
                if contacts:
                    for contact in contacts:
                        f.write(f"{contact}\n")
                else:
                    f.write("(no contacts found)\n")
                
                f.write("\n")
        
        files_written.append(filename)
        print(f"   Generated: {filename}")
    
    return files_written

def print_summary_statistics(frame_contacts):
    """Print summary statistics for the analysis"""
    
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    total_frames = len(frame_contacts)
    
    # Calculate statistics for each interaction type
    contact_types = ['A_internal', 'B_internal', 'A_B_inter']
    type_names = {
        'A_internal': 'Protein A Internal (A-A)',
        'B_internal': 'Protein B Internal (B-B)',
        'A_B_inter': 'Inter-protein (A-B)'
    }
    
    for contact_type in contact_types:
        print(f"\n{type_names[contact_type]}:")
        print("-" * 40)
        
        total_contacts = sum(len(frame_contacts[t][contact_type]) 
                           for t in frame_contacts)
        avg_contacts = total_contacts / total_frames if total_frames > 0 else 0
        
        max_contacts = max(len(frame_contacts[t][contact_type]) 
                          for t in frame_contacts)
        min_contacts = min(len(frame_contacts[t][contact_type]) 
                          for t in frame_contacts)
        
        print(f"  Total contacts across all frames: {total_contacts}")
        print(f"  Average contacts per frame: {avg_contacts:.1f}")
        print(f"  Max contacts in a frame: {max_contacts}")
        print(f"  Min contacts in a frame: {min_contacts}")
        
        # Find most common contacts
        contact_counts = defaultdict(int)
        for t in frame_contacts:
            for contact in frame_contacts[t][contact_type]:
                contact_counts[contact] += 1
        
        if contact_counts:
            print(f"\n  Top 10 most frequent contacts:")
            sorted_contacts = sorted(contact_counts.items(), 
                                   key=lambda x: x[1], reverse=True)
            for i, (contact, count) in enumerate(sorted_contacts[:10], 1):
                persistence = (count / total_frames) * 100
                print(f"    {i}. {contact}: {count} frames ({persistence:.1f}%)")
    
    # Time range
    if frame_contacts:
        times = sorted(frame_contacts.keys())
        print(f"\nTime range: {times[0]:.1f} - {times[-1]:.1f} ns")
        print(f"Total frames analyzed: {total_frames}")

def main():
    """Main analysis pipeline for dimer system"""
    print("="*70)
    print("DIMER PROTEIN CONTACT ANALYSIS")
    print("="*70)
    print(f"Configuration:")
    print(f"  Distance cutoff: {cutoff} Å")
    print(f"  Chain size: {chain_size} residues each")
    print(f"  Chain A indices: {CHAIN_A_START}-{CHAIN_A_END}")
    print(f"  Chain B indices: {CHAIN_B_START}-{CHAIN_B_END}")
    print(f"  Fold: {FOLD}")
    print("\nAnalyzing three types of interactions:")
    print("  1. Protein A internal (A-A)")
    print("  2. Protein B internal (B-B)")
    print("  3. Inter-protein (A-B)")
    
    # Process each pH condition
    for condition in pH_conditions:
        PH = condition['PH']
        PHdot = condition['PHdot']
        
        print(f"\n{'='*70}")
        print(f"Processing pH {PHdot}")
        print(f"{'='*70}")
        
        # File paths
        pdb_file = f"FOLD{FOLD}_pH{PH}_ion_1ns.pdb"
        
        if not os.path.exists(pdb_file):
            print(f"  ERROR: PDB file not found: {pdb_file}")
            continue
        
        # Create residue mapping
        print("  Creating residue mapping...")
        residue_mapping = create_residue_mapping(pdb_file)
        
        # Process all frames
        print("  Analyzing contacts for every frame...")
        frame_contacts = process_pdb_contacts_per_frame(pdb_file, residue_mapping)
        
        if not frame_contacts:
            print(f"  No contact data generated for pH {PHdot}")
            continue
        
        # Save results
        save_contact_files(frame_contacts, FOLD, PH, PHdot)
        
        # Print statistics
        print_summary_statistics(frame_contacts)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nGenerated files for each pH condition:")
    print("  - FOLD{X}_pH{Y}_ProtA_internal_interactions.txt")
    print("  - FOLD{X}_pH{Y}_ProtB_internal_interactions.txt")
    print("  - FOLD{X}_pH{Y}_ProtA_ProtB_inter_interactions.txt")

if __name__ == "__main__":
    main()