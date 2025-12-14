# -*- coding: utf-8 -*-
"""
Created on Sat Dec  6 20:11:01 2025

@author: useraw
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from collections import defaultdict
import os
import re
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array

# === CONFIGURATION ===
cutoff = 5.0  # distance cutoff in Å
FOLD = '1-2'

# Data residue range (what's actually in your files)
data_start_residue = 1
data_end_residue = 154

# Display residue range (what you want to show on the plots)
display_start_residue = 188
display_end_residue = 341

# NEW: Frame processing option
# Set to None for all frames, or 50 to use only last 50 frames
USE_LAST_N_FRAMES = None

residues_per_chain = data_end_residue - data_start_residue + 1
matrix_size = 2 * residues_per_chain  # for chains A and B

pH_conditions = [
    {'PH': '40', 'PHdot': '4.0'},
    {'PH': '74', 'PHdot': '7.4'},
    {'PH': '85', 'PHdot': '8.5'}
]

def residue_index(residue_id):
    """Return matrix index from residue ID like 'A_200' - using data residue range"""
    try:
        chain, resi = residue_id.split('_')
        resi = int(resi)
        if not (data_start_residue <= resi <= data_end_residue):
            return None
        offset = 0 if chain == 'A' else residues_per_chain
        return offset + (resi - data_start_residue)
    except Exception:
        return None

def load_ion_interactions(filename, expected_ion_type, last_n_frames=None):
    """Load ion interaction data from file"""
    if not os.path.exists(filename):
        print(f"❌ ERROR: File '{filename}' not found.")
        return pd.DataFrame()
    
    print(f"✅ Reading {expected_ion_type} contact file: {filename}")
    
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    print(f"📄 Total lines in file: {len(lines)}")
    
    data = []
    for line_num, line in enumerate(lines):
        if line_num < 3:
            print(f"   Line {line_num + 1}: {line}")
        
        match = re.match(r"frame\s+(\d+)\s+(\w+)\s+(\d+)\s+chain\s+(\w)\s+interacts_with_(Cl-|Na\+)_([0-9]+)", line)
        if match:
            frame, resname, resid, chain, ion_type, ion_resid = match.groups()
            residue_id = f"{chain}_{resid}"
            ion_id = f"{ion_type}_{ion_resid}"
            data.append({
                "frame": int(frame),
                "resname": resname,
                "resid": int(resid),
                "chain": chain,
                "ion_type": ion_type,
                "ion_resid": int(ion_resid),
                "ion_id": ion_id,
                "residue_id": residue_id
            })
    
    df = pd.DataFrame(data)
    
    # Filter to last N frames if specified
    if last_n_frames is not None and len(df) > 0:
        max_frame = df['frame'].max()
        min_frame_to_keep = max_frame - last_n_frames + 1
        df = df[df['frame'] >= min_frame_to_keep]
        print(f"🔍 Filtered to last {last_n_frames} frames (frames {min_frame_to_keep}-{max_frame})")
    
    print(f"✅ Loaded {len(df)} {expected_ion_type} interactions from file")
    
    if len(df) > 0:
        unique_ions = df['ion_id'].nunique()
        unique_residues = df['residue_id'].nunique()
        unique_frames = df['frame'].nunique()
        print(f"   📊 Stats: {unique_ions} unique ions, {unique_residues} unique residues, {unique_frames} frames")
        
        residue_numbers = [int(rid.split('_')[1]) for rid in df['residue_id']]
        print(f"   📍 Residue range: {min(residue_numbers)} - {max(residue_numbers)}")
    
    return df

def calculate_ion_contacts(df):
    """Calculate ion-mediated contacts between residues"""
    print(f"🔗 Calculating ion-mediated contacts...")
    
    contacts = defaultdict(int)
    
    for frame, frame_group in df.groupby("frame"):
        for ion_id, ion_group in frame_group.groupby("ion_id"):
            residues = sorted(ion_group['residue_id'].unique())
            
            if len(residues) >= 2:
                print(f"   🧲 Frame {frame}, Ion {ion_id}: {len(residues)} residues ({residues})")
            
            if len(residues) < 2:
                continue
                
            for r1, r2 in combinations(residues, 2):
                key = tuple(sorted((r1, r2)))
                contacts[key] += 1
    
    print(f"   📊 Found {len(contacts)} unique ion-mediated contact pairs")
    print(f"   📊 Total ion-mediated contacts: {sum(contacts.values())}")
    
    return contacts

def calculate_direct_ion_interactions(df):
    """Count direct residue-ion interactions (simpler metric)"""
    print(f"🎯 Calculating direct residue-ion interactions...")
    
    contact_counts = defaultdict(int)
    for _, row in df.iterrows():
        residue_id = row["residue_id"]
        contact_counts[residue_id] += 1
    
    print(f"   📊 Found {len(contact_counts)} residues with ion interactions")
    print(f"   📊 Total residue-ion interactions: {sum(contact_counts.values())}")
    
    return contact_counts

def ion_contacts_to_array(contact_dict):
    """Convert ion contact dictionary to numpy array"""
    ion_array = np.zeros(matrix_size, dtype=int)
    for residue_id, count in contact_dict.items():
        idx = residue_index(residue_id)
        if idx is not None:
            ion_array[idx] = count
    return ion_array

def contacts_to_matrix(contact_dict):
    matrix = np.zeros((matrix_size, matrix_size), dtype=int)
    for (r1, r2), count in contact_dict.items():
        i = residue_index(r1)
        j = residue_index(r2)
        if i is not None and j is not None:
            matrix[i, j] = matrix[j, i] = count
    return matrix

def process_pdb_single_pass(pdb_filename, last_n_frames=None):
    if not os.path.exists(pdb_filename):
        print(f"❌ ERROR: PDB file '{pdb_filename}' not found.")
        return np.zeros((matrix_size, matrix_size), dtype=int), np.zeros((matrix_size, 1), dtype=int)

    print(f"✅ Loading structure: {pdb_filename}")
    u = mda.Universe(pdb_filename, format='PDB')
    n_frames = len(u.trajectory)
    
    # Determine which frames to process
    if last_n_frames is not None and last_n_frames < n_frames:
        start_frame = n_frames - last_n_frames
        frames_to_process = range(start_frame, n_frames)
        print(f"🔍 Processing last {last_n_frames} frames (frames {start_frame}-{n_frames-1})")
    else:
        frames_to_process = range(n_frames)
        print(f"✅ Found {n_frames} frames - processing all in single pass")

    contact_counts = np.zeros((matrix_size, matrix_size), dtype=int)
    time_matrix = np.zeros((matrix_size, len(frames_to_process)), dtype=int)

    for time_idx, frame_idx in enumerate(frames_to_process):
        if time_idx % 10 == 0:
            print(f"Processing frame {frame_idx + 1}/{n_frames} (position {time_idx + 1}/{len(frames_to_process)})")

        u.trajectory[frame_idx]
        
        try:
            protein_atoms = u.select_atoms("protein and not name H*")
        except:
            protein_atoms = u.select_atoms("not name H*")
        
        if len(protein_atoms) == 0:
            print(f"❌ No protein atoms found in frame {frame_idx}")
            continue

        residue_ids = []
        for atom in protein_atoms:
            if hasattr(atom, 'chainid') and atom.chainid.strip():
                chain = atom.chainid.strip()
            elif hasattr(atom, 'segid') and atom.segid.strip():
                chain = atom.segid.strip()
            else:
                chain = 'A' if atom.resid < 200 else 'B'
            
            residue_ids.append(f"{chain}_{atom.resid}")

        positions = protein_atoms.positions
        dist_matrix = distance_array(positions, positions)
        contact_matrix = dist_matrix < cutoff

        frame_contacts = np.zeros((matrix_size, matrix_size), dtype=bool)
        residues_in_contact = set()

        atom_contacts = np.where(contact_matrix)
        for i, j in zip(atom_contacts[0], atom_contacts[1]):
            if i != j:
                res_i = residue_ids[i]
                res_j = residue_ids[j]
                idx_i = residue_index(res_i)
                idx_j = residue_index(res_j)
                if idx_i is not None and idx_j is not None and idx_i != idx_j:
                    frame_contacts[idx_i, idx_j] = True
                    frame_contacts[idx_j, idx_i] = True
                    residues_in_contact.update([idx_i, idx_j])

        contact_counts += frame_contacts.astype(int)
        for res in residues_in_contact:
            time_matrix[res, time_idx] = 1

    print(f"✅ Single-pass processing complete")
    return contact_counts, time_matrix

def calculate_residue_contact_counts(matrix, ion_array=None):
    """Calculate total contacts per residue"""
    if ion_array is not None:
        return ion_array
    else:
        return np.sum(matrix, axis=1)

def create_custom_colormap(base_cmap_name):
    """Create a custom colormap with white for 0 values"""
    from matplotlib.colors import ListedColormap
    
    base_cmap = plt.cm.get_cmap(base_cmap_name)
    colors = ['white'] + [base_cmap(i) for i in np.linspace(0.2, 1.0, 255)]
    
    return ListedColormap(colors)

def add_frame_to_heatmap(ax):
    """Add a frame around the heatmap"""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_edgecolor('black')

def plot_contact_maps(direct_contacts, cla_matrix, sod_matrix, PHdot, cutoff, FOLD, PH, 
                      zoom_range=None, suffix="full"):
    """Plot the three contact maps side by side
    
    Parameters:
    -----------
    zoom_range : tuple or None
        If provided as (start, end), zooms to display residues in that range.
        Example: (263, 319) for residues 263-319
    suffix : str
        Suffix for the output filename (e.g., "full" or "zoom_263-319")
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    residues_per_chain = matrix_size // 2
    
    # Calculate zoom indices if zoom_range is provided
    if zoom_range is not None:
        zoom_start, zoom_end = zoom_range
        zoom_start_idx = zoom_start - display_start_residue
        zoom_end_idx = zoom_end - display_start_residue
        
        zoom_start_idx = max(0, zoom_start_idx)
        zoom_end_idx = min(residues_per_chain - 1, zoom_end_idx)
        
        zoom_indices = list(range(zoom_start_idx, zoom_end_idx + 1)) + \
                      list(range(residues_per_chain + zoom_start_idx, residues_per_chain + zoom_end_idx + 1))
        
        direct_contacts_plot = direct_contacts[np.ix_(zoom_indices, zoom_indices)]
        cla_matrix_plot = cla_matrix[np.ix_(zoom_indices, zoom_indices)]
        sod_matrix_plot = sod_matrix[np.ix_(zoom_indices, zoom_indices)]
        
        plot_matrix_size = len(zoom_indices)
        zoom_residues_per_chain = zoom_end_idx - zoom_start_idx + 1
        
        print(f"   🔍 Zooming to residues {zoom_start}-{zoom_end} (matrix indices {zoom_start_idx}-{zoom_end_idx})")
        print(f"   📐 Zoom matrix size: {plot_matrix_size}x{plot_matrix_size}")
    else:
        direct_contacts_plot = direct_contacts
        cla_matrix_plot = cla_matrix
        sod_matrix_plot = sod_matrix
        plot_matrix_size = matrix_size
        zoom_residues_per_chain = residues_per_chain
        zoom_start_idx = 0
    
    # Configure tick positions
    if zoom_range is not None:
        tick_spacing = 5
        tick_positions = np.arange(4, zoom_residues_per_chain, tick_spacing)
    else:
        tick_spacing = 10
        tick_positions = np.arange(9, zoom_residues_per_chain, tick_spacing)
    
    # Calculate labels
    tick_labels = []
    for pos in tick_positions:
        display_residue = display_start_residue + zoom_start_idx + pos
        tick_labels.append(f"{display_residue}")
    
    tick_positions_with_b = list(tick_positions) + list(tick_positions + zoom_residues_per_chain)
    tick_labels_with_b = tick_labels * 2
    
    # Create custom colormaps
    red_cmap = create_custom_colormap('Reds')
    blue_cmap = create_custom_colormap('Blues')
    purple_cmap = create_custom_colormap('Purples')
    
    # Determine scales
    direct_max = np.max(direct_contacts_plot) if np.max(direct_contacts_plot) > 0 else 1500
    cla_max = np.max(cla_matrix_plot) if np.max(cla_matrix_plot) > 0 else 40
    sod_max = np.max(sod_matrix_plot) if np.max(sod_matrix_plot) > 0 else 10
    
    direct_scale = min(1500, int(np.ceil(direct_max / 100) * 100))
    cla_scale = min(40, max(10, int(np.ceil(cla_max / 5) * 5)))
    sod_scale = min(10, max(5, int(np.ceil(sod_max / 2) * 2)))
    
    # Plot 1: Direct Contact Map
    direct_ticks = np.arange(0, direct_scale + 1, direct_scale // 6)
    sns.heatmap(direct_contacts_plot, 
                cmap=red_cmap, square=True, 
                vmin=0, vmax=direct_scale,
                cbar_kws={'shrink': 0.8, 'aspect': 20, 'label': 'Direct Contacts', 'ticks': direct_ticks}, 
                ax=axes[0],
                xticklabels=False, yticklabels=False)
    
    axes[0].set_xticks(tick_positions_with_b)
    axes[0].set_xticklabels(tick_labels_with_b, rotation=45)
    axes[0].set_yticks(tick_positions_with_b)
    axes[0].set_yticklabels(tick_labels_with_b, rotation=0)
    axes[0].set_xlabel('Residue Number')
    axes[0].set_ylabel('Residue Number')
    
    title = f"Direct Contacts\n(cutoff={cutoff}Å, pH={PHdot})"
    if zoom_range:
        title += f"\nZoom: {zoom_range[0]}-{zoom_range[1]}"
    axes[0].set_title(title)
    axes[0].invert_yaxis()
    
    axes[0].axhline(y=zoom_residues_per_chain, color='black', linewidth=2, linestyle='--', alpha=0.8)
    axes[0].axvline(x=zoom_residues_per_chain, color='black', linewidth=2, linestyle='--', alpha=0.8)
    add_frame_to_heatmap(axes[0])
    
    # Plot 2: CLA-mediated Contact Map
    cla_ticks = np.arange(0, cla_scale + 1, max(1, cla_scale // 5))
    sns.heatmap(cla_matrix_plot, 
                cmap=blue_cmap, square=True, 
                vmin=0, vmax=cla_scale,
                cbar_kws={'shrink': 0.8, 'aspect': 20, 'label': 'Cl⁻-mediated Contacts', 'ticks': cla_ticks}, 
                ax=axes[1],
                xticklabels=False, yticklabels=False)
    
    axes[1].set_xticks(tick_positions_with_b)
    axes[1].set_xticklabels(tick_labels_with_b, rotation=45)
    axes[1].set_yticks(tick_positions_with_b)
    axes[1].set_yticklabels(tick_labels_with_b, rotation=0)
    axes[1].set_xlabel('Residue Number')
    axes[1].set_ylabel('Residue Number')
    
    title = f"Cl⁻-mediated Contacts\n(pH={PHdot})"
    if zoom_range:
        title += f"\nZoom: {zoom_range[0]}-{zoom_range[1]}"
    axes[1].set_title(title)
    axes[1].invert_yaxis()
    
    axes[1].axhline(y=zoom_residues_per_chain, color='black', linewidth=2, linestyle='--', alpha=0.8)
    axes[1].axvline(x=zoom_residues_per_chain, color='black', linewidth=2, linestyle='--', alpha=0.8)
    add_frame_to_heatmap(axes[1])
    
    # Plot 3: SOD-mediated Contact Map
    sod_ticks = np.arange(0, sod_scale + 1, max(1, sod_scale // 5))
    sns.heatmap(sod_matrix_plot, 
                cmap=purple_cmap, square=True, 
                vmin=0, vmax=sod_scale,
                cbar_kws={'shrink': 0.8, 'aspect': 20, 'label': 'Na⁺-mediated Contacts', 'ticks': sod_ticks}, 
                ax=axes[2],
                xticklabels=False, yticklabels=False)
    
    axes[2].set_xticks(tick_positions_with_b)
    axes[2].set_xticklabels(tick_labels_with_b, rotation=45)
    axes[2].set_yticks(tick_positions_with_b)
    axes[2].set_yticklabels(tick_labels_with_b, rotation=0)
    axes[2].set_xlabel('Residue Number')
    axes[2].set_ylabel('Residue Number')
    
    title = f"Na⁺-mediated Contacts\n(pH={PHdot})"
    if zoom_range:
        title += f"\nZoom: {zoom_range[0]}-{zoom_range[1]}"
    axes[2].set_title(title)
    axes[2].invert_yaxis()
    
    axes[2].axhline(y=zoom_residues_per_chain, color='black', linewidth=2, linestyle='--', alpha=0.8)
    axes[2].axvline(x=zoom_residues_per_chain, color='black', linewidth=2, linestyle='--', alpha=0.8)
    add_frame_to_heatmap(axes[2])
    
    plt.tight_layout()
    
    # Save the figure
    combined_output_img = f"FOLD{FOLD}_pH{PH}_combined_contact_maps_{cutoff}A_{suffix}.png"
    current_dir = os.getcwd()
    full_path = os.path.join(current_dir, combined_output_img)
    
    try:
        plt.savefig(full_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 Combined contact maps saved as: {full_path}")
        print(f"📁 File exists: {os.path.exists(full_path)}")
        print(f"   📈 Direct contacts max: {np.max(direct_contacts_plot)}")
        print(f"   📈 Cl⁻-mediated max: {np.max(cla_matrix_plot)}")
        print(f"   📈 Na⁺-mediated max: {np.max(sod_matrix_plot)}")
    except Exception as e:
        print(f"❌ Error saving combined contact maps: {e}")
    
    plt.close()

def main():
    print("🔬 Starting Dimer Contact Analysis Pipeline")
    print(f"📋 Configuration: cutoff={cutoff} Å, matrix_size={matrix_size}, fold={FOLD}")
    if USE_LAST_N_FRAMES is not None:
        print(f"⚠️  FRAME MODE: Processing only last {USE_LAST_N_FRAMES} frames")
    else:
        print(f"✅ FRAME MODE: Processing all frames")

    for condition in pH_conditions:
        PH = condition['PH']
        PHdot = condition['PHdot']

        print(f"\n🧪 Processing pH {PHdot}...")
        cla_file = f"FOLD{FOLD}_pH{PH}_ion_residues_with_CLA_interactions_{cutoff}A.txt"
        sod_file = f"FOLD{FOLD}_pH{PH}_ion_residues_with_SOD_interactions_{cutoff}A.txt"
        pdb_file = f"FOLD{FOLD}_pH{PH}_ion_1ns.pdb"

        # Load ion interactions
        cla_df = load_ion_interactions(cla_file, "CLA", last_n_frames=USE_LAST_N_FRAMES)
        sod_df = load_ion_interactions(sod_file, "SOD", last_n_frames=USE_LAST_N_FRAMES)

        # Calculate contacts
        print(f"\n🔬 Analyzing Cl⁻ interactions...")
        cla_mediated_contacts = calculate_ion_contacts(cla_df)
        cla_direct_interactions = calculate_direct_ion_interactions(cla_df)
        
        print(f"\n🔬 Analyzing Na⁺ interactions...")
        sod_mediated_contacts = calculate_ion_contacts(sod_df)
        sod_direct_interactions = calculate_direct_ion_interactions(sod_df)

        # Convert to matrices/arrays
        cla_matrix = contacts_to_matrix(cla_mediated_contacts)
        sod_matrix = contacts_to_matrix(sod_mediated_contacts)
        cla_array = ion_contacts_to_array(cla_direct_interactions)
        sod_array = ion_contacts_to_array(sod_direct_interactions)

        # Process direct contacts from PDB
        direct_matrix, time_matrix = process_pdb_single_pass(pdb_file, last_n_frames=USE_LAST_N_FRAMES)
        direct_counts = calculate_residue_contact_counts(direct_matrix)

        print(f"\n📈 Summary (pH {PHdot}):")
        print(f"   Direct protein-protein contacts: {np.sum(direct_matrix)//2}")
        print(f"   Ion-mediated contacts (Cl⁻): {np.sum(cla_matrix)//2}")
        print(f"   Ion-mediated contacts (Na⁺): {np.sum(sod_matrix)//2}")
        print(f"   Direct residue-Cl⁻ interactions: {np.sum(cla_array)}")
        print(f"   Direct residue-Na⁺ interactions: {np.sum(sod_array)}")
        
        # Check residue ranges
        if len(cla_df) > 0 or len(sod_df) > 0:
            all_residues = []
            if len(cla_df) > 0:
                all_residues.extend([int(rid.split('_')[1]) for rid in cla_df['residue_id']])
            if len(sod_df) > 0:
                all_residues.extend([int(rid.split('_')[1]) for rid in sod_df['residue_id']])
            
            if all_residues:
                data_min, data_max = min(all_residues), max(all_residues)
                print(f"   ✅ Data residue range: {data_min}-{data_max}, Expected: {data_start_residue}-{data_end_residue}")
                print(f"   📊 Display range: {display_start_residue}-{display_end_residue}")
        
        # Print top residues
        if np.sum(direct_counts) > 0:
            top_direct = np.argsort(direct_counts)[-10:][::-1]
            print("\n🔝 Top 10 Direct Contact Residues:")
            for rank, idx in enumerate(top_direct):
                if direct_counts[idx] > 0:
                    chain = 'A' if idx < residues_per_chain else 'B'
                    data_res_num = data_start_residue + (idx % residues_per_chain)
                    display_res_num = display_start_residue + (idx % residues_per_chain)
                    print(f"   {rank+1}. {chain}_{display_res_num} (data:{data_res_num}): {direct_counts[idx]}")
        
        if np.sum(cla_array) > 0:
            top_cla = np.argsort(cla_array)[-5:][::-1]
            print("\n🔝 Top 5 Cl⁻ Interacting Residues:")
            for rank, idx in enumerate(top_cla):
                if cla_array[idx] > 0:
                    chain = 'A' if idx < residues_per_chain else 'B'
                    data_res_num = data_start_residue + (idx % residues_per_chain)
                    display_res_num = display_start_residue + (idx % residues_per_chain)
                    print(f"   {rank+1}. {chain}_{display_res_num} (data:{data_res_num}): {cla_array[idx]}")
        
        if np.sum(sod_array) > 0:
            top_sod = np.argsort(sod_array)[-5:][::-1]
            print("\n🔝 Top 5 Na⁺ Interacting Residues:")
            for rank, idx in enumerate(top_sod):
                if sod_array[idx] > 0:
                    chain = 'A' if idx < residues_per_chain else 'B'
                    data_res_num = data_start_residue + (idx % residues_per_chain)
                    display_res_num = display_start_residue + (idx % residues_per_chain)
                    print(f"   {rank+1}. {chain}_{display_res_num} (data:{data_res_num}): {sod_array[idx]}")
        
        # Generate FULL contact maps
        print(f"\n🎨 Generating FULL contact maps...")
        plot_contact_maps(direct_matrix, cla_matrix, sod_matrix, PHdot, cutoff, FOLD, PH, 
                         zoom_range=None, suffix="full")
        
        # Generate ZOOMED contact maps (residues 263-319)
        print(f"\n🔍 Generating ZOOMED contact maps (residues 263-319)...")
        plot_contact_maps(direct_matrix, cla_matrix, sod_matrix, PHdot, cutoff, FOLD, PH,
                         zoom_range=(263, 319), suffix="zoom_263-319")

if __name__ == "__main__":
    main()