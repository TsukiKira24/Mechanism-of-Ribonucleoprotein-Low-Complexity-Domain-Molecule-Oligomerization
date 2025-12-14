# -*- coding: utf-8 -*-
"""
Violin Plot Analysis of Inter-Chain Fibril Contacts in Dimer Simulations
Analyzes how well inter-chain contacts between ProtA and ProtB 
recreate the inter-chain contacts found in the 6WQK fibril structure
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu, kruskal
import os
import re
from pathlib import Path

# Color scheme for pH conditions
ph_colors = {
    4.0: '#bfff00',    # Lime green for acidic pH 4
    7.4: '#228B22',    # Forest green for neutral pH 7.4
    8.5: '#008B8B'     # Dark cyan for alkaline pH 8.5
}

def load_fibril_reference_contacts(filename):
    """Load reference fibril inter-chain contacts from 6WQK file"""
    print(f"Loading fibril reference contacts from {filename}...")
    
    if not os.path.exists(filename):
        print(f"File {filename} not found!")
        return set()
    
    reference_contacts = set()
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Skip header lines
    start_parsing = False
    for line in lines:
        line = line.strip()
        if line.startswith('FORMAT:'):
            start_parsing = True
            continue
        
        if start_parsing and line and '[INTER-CHAIN]' in line:
            # Extract the contact part (before [INTER-CHAIN])
            contact_part = line.split('[INTER-CHAIN]')[0].strip()
            
            # Parse the contact
            res1_name, res1_num, res2_name, res2_num = parse_contact_line(contact_part)
            
            if (res1_name is not None and res2_name is not None and 
                is_valid_contact(res1_name, res1_num, res2_name, res2_num)):
                
                # Create unique contact key
                contact_key = create_unique_contact_key(res1_name, res1_num, res2_name, res2_num)
                reference_contacts.add(contact_key)
    
    print(f"Found {len(reference_contacts)} unique fibril inter-chain contacts")
    return reference_contacts

def parse_contact_line(contact_line):
    """Parse a contact line and return residue information"""
    if '-' not in contact_line:
        return None, None, None, None
    
    try:
        res1, res2 = contact_line.split('-', 1)
        
        # Extract residue name and number (handle both formats)
        # Format 1: A_ARG191 (from simulation files)
        # Format 2: ARG191 (from 6WQK file)
        match1 = re.match(r'(?:[A-Z]_)?([A-Z]{3})(\d+)', res1.strip())
        match2 = re.match(r'(?:[A-Z]_)?([A-Z]{3})(\d+)', res2.strip())
        
        if match1 and match2:
            res1_name, res1_num = match1.groups()
            res2_name, res2_num = match2.groups()
            return res1_name, int(res1_num), res2_name, int(res2_num)
        
    except:
        pass
    
    return None, None, None, None

def is_valid_contact(res1_name, res1_num, res2_name, res2_num):
    """Check if a contact is valid (not self-interaction)"""
    # Exclude self-interactions (same residue with same number)
    if res1_name == res2_name and res1_num == res2_num:
        return False
    
    return True

def create_unique_contact_key(res1_name, res1_num, res2_name, res2_num):
    """Create a unique key for a contact pair (sorted to avoid duplicates)"""
    contact1 = f"{res1_name}{res1_num}"
    contact2 = f"{res2_name}{res2_num}"
    
    # Sort to ensure A-B and B-A are treated as the same contact
    sorted_contacts = tuple(sorted([contact1, contact2]))
    return f"{sorted_contacts[0]}-{sorted_contacts[1]}"

def calculate_autocorrelation(time_series, max_lag=100):
    """Calculate autocorrelation function to determine decorrelation time"""
    time_series = np.array(time_series)
    autocorr = []
    
    for lag in range(max_lag):
        if lag >= len(time_series):
            break
        if lag == 0:
            autocorr.append(1.0)
        else:
            # Calculate correlation coefficient between series and lagged series
            if len(time_series[:-lag]) > 1 and np.var(time_series[:-lag]) > 0 and np.var(time_series[lag:]) > 0:
                corr = np.corrcoef(time_series[:-lag], time_series[lag:])[0, 1]
                if np.isfinite(corr):
                    autocorr.append(corr)
                else:
                    autocorr.append(0.0)
            else:
                autocorr.append(0.0)
    
    return np.array(autocorr)

def find_decorrelation_time(autocorr, threshold=0.1):
    """Find the lag at which autocorrelation drops below threshold"""
    decorr_time = 1
    for i, corr in enumerate(autocorr):
        if abs(corr) < threshold:
            decorr_time = max(1, i)  # At least 1 frame spacing
            break
    return decorr_time

def systematic_subsample(data, times, decorrelation_time):
    """Systematically subsample data every decorrelation_time frames"""
    if decorrelation_time <= 1:
        return data, times
    
    subsampled_data = []
    subsampled_times = []
    
    for i in range(0, len(data), decorrelation_time):
        subsampled_data.append(data[i])
        subsampled_times.append(times[i])
    
    return subsampled_data, subsampled_times

def parse_inter_chain_file_with_autocorr(filename, reference_contacts, last_ns_only=50):
    """Parse inter-chain contact files with autocorrelation analysis and systematic subsampling"""
    print(f"Parsing {filename} for fibril-matching inter-chain contacts...")
    
    if not os.path.exists(filename):
        print(f"File {filename} not found!")
        return []
    
    all_contacts_per_frame = []
    all_frame_times = []
    current_frame_contacts = set()
    current_time = None
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # First pass: collect all frame data
    for line in lines:
        line = line.strip()
        
        if line.startswith('t=') and 'ns' in line:
            # Save previous frame data
            if current_time is not None:
                matching_contacts = current_frame_contacts.intersection(reference_contacts)
                all_contacts_per_frame.append(len(matching_contacts))
                all_frame_times.append(current_time)
            
            # Extract time from line like "t= 5 ns"
            time_match = re.search(r't=\s*(\d+(?:\.\d+)?)\s*ns', line)
            if time_match:
                current_time = float(time_match.group(1))
            current_frame_contacts = set()
            
        elif line and not line.startswith('t=') and not line.startswith('(no contacts'):
            if '-' in line:
                res1_name, res1_num, res2_name, res2_num = parse_contact_line(line)
                
                if (res1_name is not None and res2_name is not None and 
                    is_valid_contact(res1_name, res1_num, res2_name, res2_num)):
                    
                    contact_key = create_unique_contact_key(res1_name, res1_num, res2_name, res2_num)
                    current_frame_contacts.add(contact_key)
    
    # Don't forget the last frame
    if current_time is not None:
        matching_contacts = current_frame_contacts.intersection(reference_contacts)
        all_contacts_per_frame.append(len(matching_contacts))
        all_frame_times.append(current_time)
    
    # Filter to last N nanoseconds only
    if last_ns_only and all_frame_times:
        max_time = max(all_frame_times)
        cutoff_time = max_time - last_ns_only
        
        filtered_contacts = []
        filtered_times = []
        for contact_count, time_point in zip(all_contacts_per_frame, all_frame_times):
            if time_point >= cutoff_time:
                filtered_contacts.append(contact_count)
                filtered_times.append(time_point)
        
        all_contacts_per_frame = filtered_contacts
        all_frame_times = filtered_times
        
        print(f"   Filtered to last {last_ns_only} ns: {len(all_contacts_per_frame)} frames")
        if all_frame_times:
            print(f"   Time range: {min(all_frame_times):.1f} - {max(all_frame_times):.1f} ns")
    
    if len(all_contacts_per_frame) < 10:
        print(f"   Warning: Only {len(all_contacts_per_frame)} frames available")
        return all_contacts_per_frame
    
    # Calculate autocorrelation
    print("   Calculating autocorrelation function...")
    autocorr = calculate_autocorrelation(all_contacts_per_frame, max_lag=min(100, len(all_contacts_per_frame)//2))
    decorr_time = find_decorrelation_time(autocorr, threshold=0.1)
    
    print(f"   Decorrelation time: {decorr_time} frames")
    print(f"   Autocorrelation at lag 1: {autocorr[1]:.3f}")
    print(f"   Autocorrelation at decorr time: {autocorr[min(decorr_time, len(autocorr)-1)]:.3f}")
    
    # Apply systematic subsampling
    subsampled_contacts, subsampled_times = systematic_subsample(
        all_contacts_per_frame, all_frame_times, decorr_time)
    
    print(f"   Systematic subsampling: {len(subsampled_contacts)} samples from {len(all_contacts_per_frame)} frames")
    print(f"   Effective sample reduction: {len(all_contacts_per_frame)/len(subsampled_contacts):.1f}x")
    
    if subsampled_contacts:
        print(f"   Fibril-matching contact range: {min(subsampled_contacts)} - {max(subsampled_contacts)}")
        print(f"   Average: {np.mean(subsampled_contacts):.1f} fibril-matching contacts per frame")
        total_possible = len(reference_contacts)
        max_recreated = max(subsampled_contacts) if subsampled_contacts else 0
        print(f"   Recreation efficiency: {max_recreated}/{total_possible} ({max_recreated/total_possible*100:.1f}% of fibril contacts)")
    
    return subsampled_contacts

def extract_file_info(filename):
    """Extract pH and fold information from inter-chain filename"""
    # Pattern: FOLD1_pH40_ProtA_ProtB_inter_interactions.txt
    pattern = r'FOLD(\d+)_pH(\d+)_ProtA_ProtB_inter_interactions\.txt'
    match = re.search(pattern, filename)
    
    if match:
        fold = int(match.group(1))
        ph_code = int(match.group(2))
        
        # Convert pH code to actual pH value
        ph_map = {40: 4.0, 74: 7.4, 85: 8.5}
        ph_value = ph_map.get(ph_code, ph_code/10.0)
        
        return fold, ph_value
    
    return None, None

def collect_all_inter_chain_data(last_ns_only=600):
    """Collect fibril-matching inter-chain contact data from dimer simulations"""
    
    # First, load the fibril reference contacts
    print("\nLoading fibril reference contacts...")
    reference_contacts = load_fibril_reference_contacts('6WQK_interactions_5p0A.txt')
    
    if not reference_contacts:
        print("No fibril reference contacts found! Cannot proceed with analysis.")
        return pd.DataFrame()
    
    data_records = []
    
    # Look for inter-chain time series files in current directory
    time_series_files = []
    for file in Path('.').glob('*_ProtA_ProtB_inter_interactions.txt'):
        time_series_files.append(str(file))
    
    print(f"\nFound {len(time_series_files)} inter-chain time series files:")
    for file in sorted(time_series_files):
        print(f"   {file}")
    
    # Process each file
    for filename in sorted(time_series_files):
        fold, ph_value = extract_file_info(filename)
        
        if fold is None or ph_value is None:
            print(f"Could not extract info from {filename}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing {filename}")
        print(f"FOLD{fold}, pH {ph_value}")
        print(f"{'='*60}")
        
        # Parse file with autocorrelation analysis
        contacts_per_frame = parse_inter_chain_file_with_autocorr(
            filename, reference_contacts, last_ns_only)
        
        # Add each frame as a separate record
        for contact_count in contacts_per_frame:
            data_records.append({
                'contacts': contact_count,
                'pH': ph_value,
                'fold': fold,
                'source': f'FOLD{fold}_pH{ph_value}'
            })
    
    df = pd.DataFrame(data_records)
    
    if not df.empty:
        total_fibril_contacts = len(reference_contacts)
        print(f"\n{'='*60}")
        print(f"Analysis Summary:")
        print(f"{'='*60}")
        print(f"   Total fibril reference contacts: {total_fibril_contacts}")
        print(f"   Data points collected: {len(df)}")
        print(f"   Analysis period: Last {last_ns_only} ns only")
        print("   Systematic subsampling based on autocorrelation analysis")
        print("   Inter-chain contacts (ProtA ↔ ProtB)")
        
        # Summary by pH
        print("\nSummary by pH:")
        for ph in sorted(df['pH'].unique()):
            ph_data = df[df['pH'] == ph]['contacts']
            mean_recreated = ph_data.mean()
            max_recreated = ph_data.max()
            std_recreated = ph_data.std()
            n_samples = len(ph_data)
            print(f"   pH {ph}: {mean_recreated:.1f}±{std_recreated:.1f}, Max {max_recreated}, N={n_samples}")
    
    return df

def perform_statistical_analysis(df):
    """Perform statistical analysis between pH conditions"""
    print(f"\n{'='*60}")
    print(f"Statistical Analysis:")
    print(f"{'='*60}")
    
    if df.empty:
        print("No data for analysis")
        return {}
    
    # Get pH groups
    ph_values = sorted(df['pH'].unique())
    
    if len(ph_values) < 2:
        print("Need at least 2 pH conditions for comparison")
        return {}
    
    # Kruskal-Wallis test (overall difference)
    groups = [df[df['pH'] == ph]['contacts'].values for ph in ph_values]
    kruskal_stat, kruskal_p = kruskal(*groups)
    print(f"Kruskal-Wallis test: H = {kruskal_stat:.3f}, p = {kruskal_p:.4f}")
    
    # Print descriptive statistics
    print("\nDescriptive Statistics:")
    print("-" * 60)
    for ph in ph_values:
        ph_data = df[df['pH'] == ph]['contacts']
        print(f"pH {ph}: Mean = {ph_data.mean():.1f}, Median = {ph_data.median():.1f}, "
              f"SD = {ph_data.std():.1f}, N = {len(ph_data)}")
    
    # Pairwise Mann-Whitney U tests
    pairwise_results = {}
    print("\nPairwise Comparisons (Mann-Whitney U):")
    print("-" * 60)
    
    for i, ph1 in enumerate(ph_values):
        for j, ph2 in enumerate(ph_values):
            if i < j:
                group1 = df[df['pH'] == ph1]['contacts']
                group2 = df[df['pH'] == ph2]['contacts']
                
                if len(group1) > 0 and len(group2) > 0:
                    u_stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
                    pairwise_results[(ph1, ph2)] = p_value
                    
                    # Calculate effect size (rank-biserial correlation)
                    n1, n2 = len(group1), len(group2)
                    r = 1 - (2 * u_stat) / (n1 * n2)
                    effect_size = abs(r)
                    
                    # Determine effect size magnitude
                    if effect_size < 0.1:
                        effect_mag = "negligible"
                    elif effect_size < 0.3:
                        effect_mag = "small"
                    elif effect_size < 0.5:
                        effect_mag = "medium"
                    else:
                        effect_mag = "large"
                    
                    print(f"pH {ph1} vs pH {ph2}: p = {p_value:.4f}, r = {effect_size:.3f} ({effect_mag})")
    
    return pairwise_results

def add_stat_annotation(ax, x1, x2, y, p_value, height_offset=2):
    """Add statistical annotation between two groups"""
    if p_value < 0.001:
        sig_symbol = '***'
    elif p_value < 0.01:
        sig_symbol = '**'
    elif p_value < 0.05:
        sig_symbol = '*'
    else:
        sig_symbol = 'ns'
    
    # Draw the bar
    ax.plot([x1, x1, x2, x2], [y, y + height_offset, y + height_offset, y], 
            lw=1.2, c='black')
    
    # Add the significance symbol
    ax.text((x1 + x2) * 0.5, y + height_offset, sig_symbol, 
            ha='center', va='bottom', fontsize=12, fontweight='bold')

def create_violin_plot(df):
    """Create violin plot of inter-chain contacts across pH conditions"""
    
    if df.empty:
        print("No data to plot!")
        return
    
    # Create the plot
    try:
        plt.figure(figsize=(3, 5))
        ax = plt.gca()
        
        # Get pH values and sort them
        ph_values = sorted(df['pH'].unique())
        
        # Create violin plot
        parts = ax.violinplot([df[df['pH'] == ph]['contacts'].values for ph in ph_values],
                             positions=range(len(ph_values)),
                             widths=0.7,
                             showmeans=True,
                             showmedians=True)
        
        # Color the violins according to pH
        for i, (pc, ph) in enumerate(zip(parts['bodies'], ph_values)):
            pc.set_facecolor(ph_colors.get(ph, '#gray'))
            pc.set_alpha(0.7)
            pc.set_edgecolor("black")
            pc.set_linewidth(1)
        
        # Style the violin plot elements
        parts['cmeans'].set_color('red')
        parts['cmeans'].set_linewidth(2)
        parts['cmedians'].set_color('black')
        parts['cmedians'].set_linewidth(2)
        parts['cbars'].set_color('black')
        parts['cmins'].set_color('black')
        parts['cmaxes'].set_color('black')
        
        # Add individual points with jitter - colored by FOLD
        for i, ph in enumerate(ph_values):
            ph_data = df[df['pH'] == ph]['contacts'].values
            
            # Create jittered x positions
            x_jitter = np.random.normal(i, 0.1, len(ph_data))
            
            # Get fold information
            folds = df[df['pH'] == ph]['fold'].values
            
            # Define colors and markers for each FOLD
            fold_styles = {
                1: {'color': 'darkorange', 'marker': 'o', 'label': 'FOLD1'},
                2: {'color': 'royalblue', 'marker': 's', 'label': 'FOLD2'}
            }

            for fold in sorted(df['fold'].unique()):
                fold_mask = folds == fold
                if np.any(fold_mask):
                    style = fold_styles.get(fold, {'color': 'gray', 'marker': 'o', 'label': f'FOLD{fold}'})
                    ax.scatter(x_jitter[fold_mask], ph_data[fold_mask], 
                              c=style['color'], 
                              marker=style['marker'], s=20, alpha=0.6, 
                              edgecolors='black', linewidth=0.5,
                              label=style['label'] if i == 0 else "")
        
        # Perform statistical analysis and add annotations
        pairwise_results = perform_statistical_analysis(df)
        
        # Add statistical annotations
        if len(ph_values) >= 2:
            y_max = df['contacts'].max()
            annotation_height = y_max * 1.15
            
            # Define annotation positions
            annotation_configs = []
            
            if 4.0 in ph_values and 7.4 in ph_values:
                x1, x2 = ph_values.index(4.0), ph_values.index(7.4)
                annotation_configs.append((x1, x2, annotation_height, (4.0, 7.4)))
            
            if 4.0 in ph_values and 8.5 in ph_values:
                x1, x2 = ph_values.index(4.0), ph_values.index(8.5)
                annotation_configs.append((x1, x2, annotation_height + y_max * 0.16, (4.0, 8.5)))
            
            if 7.4 in ph_values and 8.5 in ph_values:
                x1, x2 = ph_values.index(7.4), ph_values.index(8.5)
                annotation_configs.append((x1, x2, annotation_height + y_max * 0.08, (7.4, 8.5)))
            
            # Add annotations
            for x1, x2, height, ph_pair in annotation_configs:
                if ph_pair in pairwise_results:
                    p_value = pairwise_results[ph_pair]
                    add_stat_annotation(ax, x1, x2, height, p_value, y_max * 0.01)
        
        # Customize the plot
        #ax.set_xlabel('pH Condition', fontsize=14, fontweight='bold')
        ax.set_ylabel('A-B Chains Fibril Contacts Recreated (counts)', fontsize=12, fontweight='bold')
        
        # Set x-axis
        ax.set_xticks(range(len(ph_values)))
        ax.set_xticklabels([f'pH {ph}' for ph in ph_values])
        
        # Add legend
        ax.legend(loc='upper right', frameon=True, fancybox=True)
        
        # Set y-axis limits and ticks for discrete data (0, 1, 2, ...)
        y_min = 0
        y_max_data = df['contacts'].max()
        
        # If data is binary (only 0 and 1), set appropriate y-axis
        if y_max_data <= 1:
            ax.set_ylim(-0.1, 1.6)
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['0', '1'])
        else:
            # For count data, use integer ticks
            y_max_plot = y_max_data + 1
            ax.set_ylim(-0.5, y_max_plot)
            ax.set_yticks(range(0, int(y_max_data) + 2))
        
        # Add horizontal grid lines only
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        plt.tight_layout()
        
        # Save the plot
        output_filename = 'Dimer_InterChain_Fibril_Contacts_Violin_Plot.png'
        try:
            plt.savefig(output_filename, dpi=300, bbox_inches='tight')
            print(f"\nPlot saved as {output_filename}")
        except PermissionError:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            alt_filename = f'Dimer_InterChain_Fibril_Contacts_Violin_Plot_{timestamp}.png'
            plt.savefig(alt_filename, dpi=300, bbox_inches='tight')
            print(f"Original file was locked, saved as: {alt_filename}")
        
        # Try to save PDF version
        try:
            pdf_filename = output_filename.replace('.png', '.pdf')
            plt.savefig(pdf_filename, bbox_inches='tight')
            print(f"PDF saved as {pdf_filename}")
        except:
            pass 
        
        plt.show()
        
    except Exception as e:
        print(f"Error creating plot: {e}")
        import traceback
        traceback.print_exc()

def generate_summary_statistics(df):
    """Generate and save summary statistics"""
    
    if df.empty:
        return
    
    # Create summary table
    summary_stats = []
    
    for ph in sorted(df['pH'].unique()):
        ph_data = df[df['pH'] == ph]
        
        for fold in sorted(ph_data['fold'].unique()):
            fold_data = ph_data[ph_data['fold'] == fold]['contacts']
            
            summary_stats.append({
                'pH': ph,
                'Fold': fold,
                'N_frames': len(fold_data),
                'Mean': fold_data.mean(),
                'Median': fold_data.median(),
                'Std': fold_data.std(),
                'Min': fold_data.min(),
                'Max': fold_data.max(),
                'Q25': fold_data.quantile(0.25),
                'Q75': fold_data.quantile(0.75)
            })
    
    summary_df = pd.DataFrame(summary_stats)
    
    # Save to file
    summary_filename = 'Dimer_InterChain_Fibril_Contacts_Summary_Statistics.csv'
    
    try:
        summary_df.to_csv(summary_filename, index=False, float_format='%.2f')
        print(f"Summary statistics saved to {summary_filename}")
    except PermissionError:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        alt_filename = f'Dimer_InterChain_Fibril_Contacts_Summary_Statistics_{timestamp}.csv'
        
        try:
            summary_df.to_csv(alt_filename, index=False, float_format='%.2f')
            print(f"Original file was locked, saved as: {alt_filename}")
        except Exception as e:
            print(f"Could not save CSV file: {e}")
    
    # Print to console
    print("\nSummary Statistics:")
    print(summary_df.to_string(index=False, float_format='{:.2f}'.format))

def main():
    """Main analysis pipeline for dimer inter-chain fibril contact recreation analysis"""
    print("="*60)
    print("Dimer Inter-Chain Fibril Contact Recreation Analysis")
    print("="*60)
    print("Analysis approach:")
    print("  • Load fibril inter-chain contacts from 6WQK as reference")
    print("  • Count how many fibril contacts are recreated as INTER-CHAIN")
    print("    contacts between ProtA and ProtB in dimer simulations")
    print("  • Analyze LAST 600 ns only (equilibrated region)")
    print("  • Autocorrelation analysis to determine decorrelation time")
    print("  • Systematic subsampling every decorrelation_time frames")
    print("  • Excludes self-interactions (e.g., GLY263-GLY263)")
    print("  • Counts only unique contact pairs (A-B = B-A)")
    print("  • Statistical analysis with Mann-Whitney U tests")
    print("="*60)
    
    # Collect time series data with autocorrelation-based subsampling
    print("\nCollecting fibril-matching inter-chain contact data...")
    df = collect_all_inter_chain_data(last_ns_only=600)
    
    if df.empty:
        print("\nNo time series data found!")
        print("Expected files: FOLD#_pH##_ProtA_ProtB_inter_interactions.txt")
        print("Expected reference: 6WQK_interactions_5p0A.txt")
        return
    
    print(f"\nCollected {len(df)} data points from {len(df['source'].unique())} FOLD/pH combinations")
    
    # Generate summary statistics
    print("\nGenerating summary statistics...")
    generate_summary_statistics(df)
    
    # Create violin plot
    print("\nCreating violin plot...")
    create_violin_plot(df)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)
    print("Generated files:")
    print("  • Dimer_InterChain_Fibril_Contacts_Violin_Plot.png")
    print("  • Dimer_InterChain_Fibril_Contacts_Violin_Plot.pdf") 
    print("  • Dimer_InterChain_Fibril_Contacts_Summary_Statistics.csv")


if __name__ == "__main__":
    main()