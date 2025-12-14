# Analysis Scripts Overview

## 📁 Contact-maps

Method: Ion-mediated and residue–residue interactions were extracted from the trajectories, excluding self-interactions and duplicated residues pairs. Contacts were defined as heavy-atom distances ≤ 5.0 Å, and evaluated from frames extracted at 1 ns intervals. Additionally analyzed simulations originated from two distinct initial conformations (folds) that were merged for each pH level. Interaction maps and residue-level bar plots were generated separately for residue–residue, Cl⁻–residue, and Na⁺–residue interactions.

📁 Contact-maps-monomer
### 1. Generation of files used for contact map (protein-ion) analysis
- `Extract_protein-ion_interactions.py`

### 2. Generation of contact maps and barplots
- **Supplementary:** `Contact_maps_and_barplots_for_LCD_domain.py`
- **Figure 3A-C; Supplementary:** `Contact_map_and_barplots_highlight_fragment-263-319.py`
- **Figure 3D:** `6WQK_fibril_inner_contacts_contact_map.py`

📁 Contact-maps-dimer

---

## 📁 Fibril-contacts-recreation-analysis

Method: Inter-chain contacts from the fibril crystal structure (PDB: 6WQK) were extracted as the reference for fibril-stabilizing interactions. Contacts were defined as residue pairs within 5.0 Å distance, excluding same residue-residue contacts and duplicates, and evaluated from frames extracted at 1 ns intervals. Only the final 600 ns of each simulation was analyzed to focus on equilibrated conformational states. For each simulation frame, the number of contacts matching the fibril reference set was quantified. To account for temporal correlations, the autocorrelation function of the contact counts was computed, and a decorrelation time was determined. Contact data were then systematically subsampled at intervals corresponding to this decorrelation time to ensure statistically independent samples. Violin plots were generated to visualize the distribution of fibril-matching contacts across different pH conditions and simulation starting conformation (folds). Horizontal lines indicate mean (red) and median (black). Whiskers show the full data range, and the violin shape represents the distribution and median of the values. Statistical differences between conditions were assessed using Kruskal-Wallis tests followed by pairwise Mann-Whitney U tests. Statistical comparisons were annotated directly on plots using significance brackets (*** p < 0.001, ** p < 0.01, * p < 0.05, ns = not significant). 

📁 Fibril-contacts-recreation-analysis-monomer

### 1. Generation of file with listed all fibril interactions
Based on 6WQK.pdb file:
- `6WQK_fibril_interaction_listing_generation.py`

### 2. Generation of file with listed all interactions over time
Based on MD results pdb file:
- `MD_LCD_interaction_listing_over_time_generation.py`

### 3. Generation of violin plot
- **Figure 3E:** `Fibril_contacts_recreation_subsampling_with_replacment.py`

📁 Fibril-contacts-recreation-analysis-dimer

---

## 📁 Protein-ion_interaction_analysis

Method: For each frame, the number of unique ion–protein contacts was obtained. Contacts were defined as heavy-atom distances ≤ 5.0 Å, and evaluated from frames extracted at 1 ns intervals. Violin plots were generated to visualize the distribution of ion interactions with residues across different pH conditions and simulation starting conformation (folds). Whiskers show the full data range, and the violin shape represents the distribution and median of the values. Statistical differences between conditions were assessed using Kruskal-Wallis tests followed by pairwise Mann-Whitney U tests. Statistical comparisons were annotated directly on plots using significance brackets (*** p < 0.001, ** p < 0.01, * p < 0.05, ns = not significant). 

### 1. Generation of file with listed all interactions over time
Based on MD results pdb file:
- `MD_Protein-ion_interaction_listing_generation.py`

### 2. Generation of violin plot
- **Supplementary:** `protein-ion_interation_count.py`

---

## Usage



## Requirements

- Python 3.11.12
- Required packages vary by script (some scripts require the MDAnalysis package)

## Citation

If you use these scripts, please cite: [...]
