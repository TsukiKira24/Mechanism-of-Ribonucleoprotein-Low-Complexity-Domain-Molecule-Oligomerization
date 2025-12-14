# -*- coding: utf-8 -*-
"""
Monomer Protein Interaction Analysis - Time Series Analysis
Analyzes contacts for EVERY nanosecond and saves time evolution data
MODIFIED TO TRACK INTERACTIONS OVER TIME
"""

import pandas as pd
import numpy as np
import os
import re
import MDAnalysis as mda
from MDAnalysis.analysis.distances import distance_array
from itertools import combinations
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# === CONFIGURATION ===
cutoff = 5.0  # distance cutoff in Å
FOLD = '1'

# Single chain configuration
matrix_size = 154  # Single chain size
start_residue = 188
end_residue = 341

pH_conditions = [
    {'PH': '40', 'PHdot': '4.0'},
    {'PH': '74', 'PHdot': '7.4'},
    {'PH': '85', 'PHdot': '8.5'}
]

def residue_index(residue_id):
    """Return matrix index from residue ID for single chain"""
    try:
        if '_' in residue_id:
            parts = residue_id.split('_')
            resi = int(parts[-1])
        else:
            resi = int(residue_id)
        
        idx = resi - 1
        if 0 <= idx < matrix_size:
            return idx
        return None
    except (ValueError, IndexError):
        return None

def index_to_residue_info(idx):
    """Convert matrix index back to residue info"""
    if idx < 0 or idx >= matrix_size:
        return None, None, None
    
    data_res_num = idx + 1
    display_res_num = start_residue + idx
    
    return data_res_num, display_res_num

def load_ion_interactions_with_time(filename):
    """Load ion interaction data with time information"""
    if not os.path.exists(filename):
        print(f" ERROR: File '{filename}' not found.")
        return pd.DataFrame()
    
    print(f" Reading: {filename}")
    
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    data = []
    for line in lines:
        pattern = r"frame\s+(\d+)\s+(\w+)\s+(\d+)\s+.*?interacts_with_(Cl-|Na\+)_(\d+)"
        match = re.search(pattern, line)
        
        if match:
            frame, resname, resid, ion_type, ion_resid = match.groups()
            
            residue_id = f"{resname} {resid}"
            ion_id = f"{ion_type}_{ion_resid}"
            
            # Convert frame to time (assuming 1 frame = 1 ps, so 1000 frames = 1 ns)
            time_ns = int(frame) / 1000.0
            
            data.append({
                "frame": int(frame),
                "time_ns": time_ns,
                "resname": resname,
                "resid": int(resid),
                "ion_type": ion_type,
                "ion_resid": int(ion_resid),
                "ion_id": ion_id,
                "residue_id": residue_id
            })
    
    print(f"   Loaded {len(data)} interactions")
    df = pd.DataFrame(data)
    
    if not df.empty:
        print(f"   Time range: {df['time_ns'].min():.1f} - {df['time_ns'].max():.1f} ns")
    
    return df

# Removed old functions - we're now using frame-by-frame analysis

def extract_pdb_timestamps(pdb_filename):
    """Extract timestamps from PDB file TITLE records"""
    timestamps = []
    
    if not os.path.exists(pdb_filename):
        print(f" ERROR: PDB file '{pdb_filename}' not found.")
        return timestamps
    
    with open(pdb_filename, 'r') as f:
        for line in f:
            if line.startswith('TITLE'):
                # Look for pattern like "t= 2000.00000"
                import re
                match = re.search(r't=\s*(\d+\.?\d*)', line)
                if match:
                    time_ps = float(match.group(1))
                    time_ns = time_ps / 1000.0  # Convert ps to ns
                    timestamps.append(time_ns)
    
    return timestamps

def process_pdb_contacts_per_frame(pdb_filename, residue_mapping):
    """Process PDB file and return contacts for each individual frame"""
    if not os.path.exists(pdb_filename):
        print(f" ERROR: PDB file '{pdb_filename}' not found.")
        return {}

    # First, extract timestamps from PDB headers
    timestamps = extract_pdb_timestamps(pdb_filename)
    
    u = mda.Universe(pdb_filename, format='PDB')
    n_frames = len(u.trajectory)
    
    print(f"   Found {n_frames} frames in PDB file")
    print(f"   Extracted {len(timestamps)} timestamps from headers")
    
    # If we have fewer timestamps than frames, assume regular spacing
    if len(timestamps) < n_frames:
        print(f"   Warning: Only {len(timestamps)} timestamps found for {n_frames} frames")
        print(f"   Using frame index + 1 as nanosecond number")
        timestamps = [i + 1 for i in range(n_frames)]  # 1-based ns numbering
    
    frame_contacts = {}
    
    for frame_idx, ts in enumerate(u.trajectory):
        if frame_idx % 10 == 0:
            print(f"   Processing frame {frame_idx + 1}/{n_frames}")
        
        # Get time for this frame
        if frame_idx < len(timestamps):
            frame_time = timestamps[frame_idx]
        else:
            frame_time = frame_idx + 1  # Fallback to frame-based numbering
        
        try:
            protein_atoms = u.select_atoms("protein and not name H*")
        except:
            protein_atoms = u.select_atoms("not name H*")
        
        if len(protein_atoms) == 0:
            frame_contacts[frame_time] = []
            continue

        residue_ids = protein_atoms.resids - 1
        positions = protein_atoms.positions
        
        valid_mask = (residue_ids >= 0) & (residue_ids < matrix_size)
        if not np.any(valid_mask):
            frame_contacts[frame_time] = []
            continue
            
        residue_ids = residue_ids[valid_mask]
        positions = positions[valid_mask]
        
        dist_matrix = distance_array(positions, positions)
        contact_matrix = dist_matrix < cutoff

        atom_contacts = np.where(contact_matrix)
        contacted_pairs = set()
        
        for i, j in zip(atom_contacts[0], atom_contacts[1]):
            if i != j:
                res_i = residue_ids[i]
                res_j = residue_ids[j]
                if (0 <= res_i < matrix_size and 0 <= res_j < matrix_size and 
                    res_i != res_j):
                    pair = tuple(sorted([res_i, res_j]))
                    contacted_pairs.add(pair)
        
        # Convert to residue names and store all contacts for this frame
        frame_contact_list = []
        for res_i, res_j in contacted_pairs:
            res_info_i = get_residue_display_name(res_i, residue_mapping)
            res_info_j = get_residue_display_name(res_j, residue_mapping)
            contact = f"{res_info_i}-{res_info_j}"
            frame_contact_list.append(contact)
        
        frame_contacts[frame_time] = sorted(frame_contact_list)  # Sort contacts alphabetically
    
    return frame_contacts

def get_residue_display_name(idx, residue_mapping):
    """Get display name for residue"""
    data_res_num, display_res_num = index_to_residue_info(idx)
    res_name = residue_mapping.get(idx, 'UNK')
    return f"{res_name}{display_res_num}"

def get_residue_name_from_pdb(pdb_file, residue_num):
    """Extract residue name from PDB file"""
    try:
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith('ATOM') and int(line[22:26].strip()) == residue_num:
                    return line[17:20].strip()
    except:
        pass
    return "UNK"

def create_residue_mapping(pdb_file):
    """Create mapping of matrix indices to residue names"""
    residue_mapping = {}
    
    for i in range(matrix_size):
        data_res_num = i + 1
        res_name = get_residue_name_from_pdb(pdb_file, data_res_num)
        residue_mapping[i] = res_name
    
    return residue_mapping

def format_residue_display(residue_id):
    """Convert residue_id to display format"""
    try:
        return residue_id.replace(" ", "")
    except:
        return residue_id

def save_time_series_files(time_series_data, PHdot, FOLD, PH):
    """Save time series analysis to files"""
    
    print(f"\n Saving time series files for pH {PHdot}...")
    files_written = []
    
    # 1. Overall statistics over time
    filename = f"FOLD{FOLD}_pH{PH}_TimeSeries_Overall_Stats.txt"
    with open(filename, 'w') as f:
        f.write(f"Time Series Analysis - Overall Statistics - pH {PHdot}\n")
        f.write("="*70 + "\n")
        f.write("Time_Window\tDirect_Protein\tCl_Direct\tNa_Direct\tCl_Mediated\tNa_Mediated\tTotal\n")
        
        for time_label in sorted(time_series_data.keys()):
            data = time_series_data[time_label]
            total = (data['total_direct'] + data['total_cla_direct'] + 
                    data['total_sod_direct'] + data['total_cla_mediated'] + 
                    data['total_sod_mediated'])
            
            f.write(f"{time_label}\t{data['total_direct']}\t{data['total_cla_direct']}\t"
                   f"{data['total_sod_direct']}\t{data['total_cla_mediated']}\t"
                   f"{data['total_sod_mediated']}\t{total}\n")
    
    files_written.append(filename)
    
    # 2. Direct Cl- interactions over time
    filename = f"FOLD{FOLD}_pH{PH}_TimeSeries_Cl_Direct.txt"
    with open(filename, 'w') as f:
        f.write(f"Time Series Analysis - Direct Cl- Interactions - pH {PHdot}\n")
        f.write("="*70 + "\n")
        f.write("Each residue counted once per frame only\n")
        f.write("-"*70 + "\n")
        
        # Get all unique residues that interact with Cl-
        all_cl_residues = set()
        for data in time_series_data.values():
            all_cl_residues.update(data['cla_direct'].keys())
        
        if all_cl_residues:
            # Header
            f.write("Time_Window\t" + "\t".join(sorted(all_cl_residues)) + "\tTotal\n")
            
            # Data rows
            for time_label in sorted(time_series_data.keys()):
                data = time_series_data[time_label]
                row = [time_label]
                total = 0
                for residue in sorted(all_cl_residues):
                    count = data['cla_direct'].get(residue, 0)
                    row.append(str(count))
                    total += count
                row.append(str(total))
                f.write("\t".join(row) + "\n")
        else:
            f.write("No Cl- interactions found in the trajectory.\n")
    
    files_written.append(filename)
    
    # 3. Direct Na+ interactions over time
    filename = f"FOLD{FOLD}_pH{PH}_TimeSeries_Na_Direct.txt"
    with open(filename, 'w') as f:
        f.write(f"Time Series Analysis - Direct Na+ Interactions - pH {PHdot}\n")
        f.write("="*70 + "\n")
        f.write("Each residue counted once per frame only\n")
        f.write("-"*70 + "\n")
        
        all_na_residues = set()
        for data in time_series_data.values():
            all_na_residues.update(data['sod_direct'].keys())
        
        if all_na_residues:
            f.write("Time_Window\t" + "\t".join(sorted(all_na_residues)) + "\tTotal\n")
            
            for time_label in sorted(time_series_data.keys()):
                data = time_series_data[time_label]
                row = [time_label]
                total = 0
                for residue in sorted(all_na_residues):
                    count = data['sod_direct'].get(residue, 0)
                    row.append(str(count))
                    total += count
                row.append(str(total))
                f.write("\t".join(row) + "\n")
        else:
            f.write("No Na+ interactions found in the trajectory.\n")
    
    files_written.append(filename)
    
    # 4. Direct protein-protein contacts over time (simple format)
    filename = f"FOLD{FOLD}_pH{PH}_TimeSeries_Protein_Direct_Top10.txt"
    with open(filename, 'w') as f:
        # Process each time window
        for time_label in sorted(time_series_data.keys()):
            data = time_series_data[time_label]
            
            # Extract the nanosecond number from time_label (e.g., "0-1ns" -> "1")
            ns_number = time_label.split('-')[1].replace('ns', '')
            f.write(f"{ns_number} ns\n")
            
            # Get all contacts for this time window and sort by frequency
            if data['direct_contacts']:
                sorted_contacts = sorted(data['direct_contacts'].items(), 
                                       key=lambda x: x[1], reverse=True)
                
                # Write each contact that exists in this time window
                for contact, count in sorted_contacts:
                    if count > 0:  # Only write contacts that actually occur
                        f.write(f"{contact}\n")
            
            # Add empty line after each time window for readability
            f.write("\n")
    
    files_written.append(filename)
    
    # 5. Ion-mediated contacts summary over time
    filename = f"FOLD{FOLD}_pH{PH}_TimeSeries_IonMediated_Summary.txt"
    with open(filename, 'w') as f:
        f.write(f"Time Series Analysis - Ion-Mediated Contacts Summary - pH {PHdot}\n")
        f.write("="*70 + "\n")
        
        f.write("Time_Window\tCl_Mediated_Pairs\tCl_Mediated_Total\tNa_Mediated_Pairs\tNa_Mediated_Total\n")
        
        for time_label in sorted(time_series_data.keys()):
            data = time_series_data[time_label]
            cl_pairs = len(data['cla_mediated'])
            cl_total = sum(data['cla_mediated'].values())
            na_pairs = len(data['sod_mediated'])
            na_total = sum(data['sod_mediated'].values())
            
            f.write(f"{time_label}\t{cl_pairs}\t{cl_total}\t{na_pairs}\t{na_total}\n")
    
    files_written.append(filename)
    
    # 6. Create summary of most dynamic interactions
    filename = f"FOLD{FOLD}_pH{PH}_TimeSeries_Dynamic_Analysis.txt"
    with open(filename, 'w') as f:
        f.write(f"Time Series Analysis - Dynamic Behavior Analysis - pH {PHdot}\n")
        f.write("="*70 + "\n")
        f.write("Analysis of interactions that show high variability over time\n\n")
        
        # Analyze Cl- interaction dynamics
        f.write("CL- INTERACTION DYNAMICS:\n")
        f.write("-" * 30 + "\n")
        
        cl_residue_stats = {}
        for time_label, data in time_series_data.items():
            for residue, count in data['cla_direct'].items():
                if residue not in cl_residue_stats:
                    cl_residue_stats[residue] = []
                cl_residue_stats[residue].append(count)
        
        # Calculate variance and mean for each residue
        cl_dynamics = []
        for residue, counts in cl_residue_stats.items():
            if len(counts) > 1:  # Need at least 2 time points
                mean_count = np.mean(counts)
                var_count = np.var(counts)
                max_count = np.max(counts)
                min_count = np.min(counts)
                
                # Only include residues with some activity
                if mean_count > 0.1:
                    cl_dynamics.append({
                        'residue': residue,
                        'mean': mean_count,
                        'variance': var_count,
                        'max': max_count,
                        'min': min_count,
                        'range': max_count - min_count
                    })
        
        # Sort by variance (most dynamic first)
        cl_dynamics.sort(key=lambda x: x['variance'], reverse=True)
        
        f.write("Residue\tMean\tVariance\tMax\tMin\tRange\n")
        for item in cl_dynamics[:10]:  # Top 10 most dynamic
            f.write(f"{item['residue']}\t{item['mean']:.2f}\t{item['variance']:.2f}\t"
                   f"{item['max']}\t{item['min']}\t{item['range']}\n")
        
        # Same analysis for Na+
        f.write("\nNA+ INTERACTION DYNAMICS:\n")
        f.write("-" * 30 + "\n")
        
        na_residue_stats = {}
        for time_label, data in time_series_data.items():
            for residue, count in data['sod_direct'].items():
                if residue not in na_residue_stats:
                    na_residue_stats[residue] = []
                na_residue_stats[residue].append(count)
        
        na_dynamics = []
        for residue, counts in na_residue_stats.items():
            if len(counts) > 1:
                mean_count = np.mean(counts)
                var_count = np.var(counts)
                max_count = np.max(counts)
                min_count = np.min(counts)
                
                if mean_count > 0.1:
                    na_dynamics.append({
                        'residue': residue,
                        'mean': mean_count,
                        'variance': var_count,
                        'max': max_count,
                        'min': min_count,
                        'range': max_count - min_count
                    })
        
        na_dynamics.sort(key=lambda x: x['variance'], reverse=True)
        
        f.write("Residue\tMean\tVariance\tMax\tMin\tRange\n")
        for item in na_dynamics[:10]:
            f.write(f"{item['residue']}\t{item['mean']:.2f}\t{item['variance']:.2f}\t"
                   f"{item['max']}\t{item['min']}\t{item['range']}\n")
    
    files_written.append(filename)
    
    print(f" Generated {len(files_written)} time series files:")
    for filename in files_written:
        print(f" {filename}")
    
    return files_written

def create_time_series_plots(time_series_data, PHdot, FOLD, PH):
    """Create visualization plots for time series data"""
    
    print(f"\n Creating time series plots for pH {PHdot}...")
    
    # Prepare data for plotting
    time_labels = sorted(time_series_data.keys())
    times = [time_series_data[label]['time_start'] for label in time_labels]
    
    direct_protein = [time_series_data[label]['total_direct'] for label in time_labels]
    cl_direct = [time_series_data[label]['total_cla_direct'] for label in time_labels]
    na_direct = [time_series_data[label]['total_sod_direct'] for label in time_labels]
    cl_mediated = [time_series_data[label]['total_cla_mediated'] for label in time_labels]
    na_mediated = [time_series_data[label]['total_sod_mediated'] for label in time_labels]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    plt.plot(times, direct_protein, 'b-o', label='Direct Protein-Protein', linewidth=2, markersize=4)
    plt.plot(times, cl_direct, 'r-s', label='Direct Protein-Cl⁻', linewidth=2, markersize=4)
    plt.plot(times, na_direct, 'g-^', label='Direct Protein-Na⁺', linewidth=2, markersize=4)
    plt.plot(times, cl_mediated, 'r--d', label='Cl⁻-Mediated', linewidth=2, markersize=4)
    plt.plot(times, na_mediated, 'g--v', label='Na⁺-Mediated', linewidth=2, markersize=4)
    
    plt.xlabel('Time (ns)', fontsize=12)
    plt.ylabel('Number of Interactions', fontsize=12)
    plt.title(f'Time Evolution of Protein Interactions - pH {PHdot}', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plot_filename = f"FOLD{FOLD}_pH{PH}_TimeSeries_Plot.png"
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f" Saved plot: {plot_filename}")
    
    return plot_filename

def main():
    """Main analysis pipeline for frame-by-frame analysis"""
    print(" Starting Frame-by-Frame Protein Contact Analysis")
    print(f" Configuration: cutoff={cutoff} Å, matrix_size={matrix_size}, fold={FOLD}")
    print(" Analyzing ALL protein-protein contacts in EVERY frame!")

    # Process only the specific condition you want
    PH = '40'
    PHdot = '4.0'

    print(f"\n Processing pH {PHdot}...")
    
    # File paths
    pdb_file = f"FOLD{FOLD}_pH{PH}_ion_1ns.pdb"

    # Create residue mapping
    residue_mapping = create_residue_mapping(pdb_file)

    # Process all frames and get contacts for each frame
    print(" Analyzing protein-protein contacts for every frame...")
    frame_contacts = process_pdb_contacts_per_frame(pdb_file, residue_mapping)
    
    if not frame_contacts:
        print(f" No frame contact data generated")
        return
    
    # Save the results in the exact format you want
    filename = f"FOLD{FOLD}_pH{PH}_TimeSeries_Protein_Direct_ALL_CONTACTS.txt"
    with open(filename, 'w') as f:
        # Process each frame in time order
        for frame_time in sorted(frame_contacts.keys()):
            # Write the time header (convert to integer ns if it's a whole number)
            if frame_time == int(frame_time):
                f.write(f"t= {int(frame_time)} ns\n")
            else:
                f.write(f"t= {frame_time:.1f} ns\n")
            
            # Write ALL contacts found in this frame
            contacts = frame_contacts[frame_time]
            if contacts:
                for contact in contacts:
                    f.write(f"{contact}\n")
            else:
                f.write("(no contacts found)\n")
            
            # Add empty line after each frame
            f.write("\n")
    
    print(f" Generated file: {filename}")
    
    # Print summary statistics
    total_frames = len(frame_contacts)
    total_contacts_all_frames = sum(len(contacts) for contacts in frame_contacts.values())
    avg_contacts_per_frame = total_contacts_all_frames / total_frames if total_frames > 0 else 0
    
    print(f" Analyzed {total_frames} frames")
    print(f" Total contacts across all frames: {total_contacts_all_frames}")
    print(f" Average contacts per frame: {avg_contacts_per_frame:.1f}")
    
    if frame_contacts:
        times = sorted(frame_contacts.keys())
        time_range = times[-1] - times[0]
        print(f" Time range: {times[0]:.1f} - {times[-1]:.1f} ns ({time_range:.1f} ns total)")
        
        # Find frame with most contacts
        max_contacts_frame = max(frame_contacts.items(), key=lambda x: len(x[1]))
        print(f" Frame with most contacts: t= {max_contacts_frame[0]:.1f} ns ({len(max_contacts_frame[1])} contacts)")

    print("\n Frame-by-frame analysis complete!")
    print(f"Generated: {filename}")
    print("Format: Each frame shows ALL protein-protein contacts present at that time point")

if __name__ == "__main__":
    main()
