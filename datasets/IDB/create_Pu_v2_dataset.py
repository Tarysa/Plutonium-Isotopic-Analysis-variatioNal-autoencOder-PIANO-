import pandas as pd
import os 
import numpy as np
import re
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import random
import json
from sklearn import preprocessing
from collections import defaultdict
from scipy.stats import gaussian_kde
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline, PchipInterpolator

E_common = np.linspace(0, 3.0, 4096)

# def resize_spectrum_spline(spectrum, E_min, E_max, E_common):
#     E_original = np.linspace(E_min, E_max, len(spectrum))
#     cs = CubicSpline(E_original, spectrum)
#     return cs(E_common)

def resize_spectrum_spline(spectrum, target_size):
    """Resize un spectre à target_size points via interpolation"""
    x_original = np.arange(len(spectrum))
    x_new = np.linspace(0, len(spectrum)-1, target_size)
    interp = PchipInterpolator(x_original, spectrum)
    return interp(x_new)

def parse_spectrum_file(file_path):
    """
    Parse a gamma spectrum file and extract metadata.
    
    Args:
        file_path (str): Path to the spectrum file
        
    Returns:
        dict: Dictionary containing metadata and spectrum content
    """
    try:
        with open(file_path, encoding="utf8", errors='ignore') as f:
            file_lines = f.readlines()
            
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

    metadata_dict = {}

    metadata_dict["File name"] = file_path.split("/")[-1]
    data_not_found = True
    i = 0
    while (data_not_found):
        ligne = file_lines[i]
        if "$DATA" in ligne :
            data_not_found = False 
        else:
            ligne = ligne.strip()
        if not ligne:
            continue  # ignore les lignes vides

        # Choix du séparateur
        if ":" in ligne:
            key, value = ligne.split(":", 1)
        else:
            key, value = ligne.split(None, 1)  # séparation par espace(s)

        if key == "Detector":

            detector_type = value.strip().split(" ")[0]

            if detector_type == "CZT":
            
                size_detector = value.strip().split("Size: ")[1].split()[0]
                metadata_dict[key.strip()] = f"{detector_type}_{size_detector}"
            else:
                metadata_dict[key.strip()] = f"{detector_type}"
        else:
            metadata_dict[key.strip()] = value.strip().split(" ")[0]
        i += 1

    nb_line = int(file_lines[i].split(" ")[1]) + 1

    
    if "ENER_FIT" in file_lines[i+nb_line]:
        spectrum = [0]
        #Spectrum with nb_lines values
        for j in range(nb_line - 1):
            spectrum.append(int(file_lines[i+j+1]))
        spectrum = np.array(spectrum)
        id_to_continue = i+nb_line+1

    else:
        spectrum = []
        for j in range(nb_line):
            spectrum.append(int(file_lines[i+j+1]))
        spectrum = np.array(spectrum)
        id_to_continue = i+nb_line+2

    
    spectrum = resize_spectrum_spline(spectrum, 4096)

    metadata_dict["content"] = [spectrum]
    
    metadata_dict["ENER_FIT"] = file_lines[id_to_continue].strip().split(" ")[1]
    for j in range(id_to_continue+1, len(file_lines)):
        ligne = file_lines[j]
        ligne = ligne.strip()
        if not ligne:
            continue  # ignore les lignes vides

        # Choix du séparateur
        try:
            if ":" in ligne:
                key, value = ligne.split(":", 1)
            else:
                key, value = ligne.split(None, 1)  # séparation par espace(s)

            metadata_dict[key.strip()] = value.strip().split(" ")[0]
        except Exception as e:
            continue

    key_to_remove = [
        k for k, v in metadata_dict.items()
        if v is None or v == ""
    ]

    for k in key_to_remove:
        del metadata_dict[k]

    return metadata_dict


def process_metadata(metadata_dict):
    """
    Process and normalize extracted metadata.
    
    Args:
        metadata_dict (dict): Dictionary of raw metadata
        detector_count (int): Detector identifier
        
    Returns:
        dict: Dictionary of processed metadata
    """
    # Expected columns in final dataset
    expected_columns = [
        "File name", "Detector", "238Pu", "239Pu", "240Pu", "241Pu", "242Pu", "content", "ENER_FIT"
    ]
 
    # Add detector identifier
    
    cols_PU = ["238Pu", "239Pu", "240Pu", "241Pu", "242Pu"]  # les colonnes à sommer

    sum_PU = 0
    
    # Create final dictionary with only necessary columns
    processed_dict = {}
    for col in expected_columns:
        if col in metadata_dict:
            if metadata_dict[col] != "Unknown":
                processed_dict[col] = metadata_dict[col]
                if col in cols_PU:
                    sum_PU += float(metadata_dict[col])
            else:
                processed_dict[col] = 0
        else:
            processed_dict[col] = -100

    col_PU_sums = (sum_PU > 99) and (sum_PU < 101)

    
    if col_PU_sums:
        return processed_dict
    else:
        print("Sum not equal to one")
        #1388/1591 rows 
        return None

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

def plot_row_sum_histogram(df, columns_to_sum, num_bins=20):
    temp_df = df[columns_to_sum].apply(pd.to_numeric, errors='coerce')
    # 2. On supprime les lignes qui contiendraient des NaN après conversion (optionnel)
    temp_df = temp_df.dropna()
    
    # 3. Calcul de la somme ligne par ligne
    row_sums = temp_df.sum(axis=1)
    
    plt.figure(figsize=(12, 7)) # Un peu plus large pour l'espace
    
    # Suppression de kde=True si les données sont trop disparates, 
    # ou garde-le si tu veux voir la tendance.
    ax = sns.histplot(row_sums, bins=num_bins, kde=True, color='royalblue', edgecolor='white')
    
    # --- NETTOYAGE DE L'AXE X ---
    
    # 1. On limite à 8 graduations maximum
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
    
    # 2. On formate les nombres pour éviter les .0000000002226
    # 'g' choisit le format le plus compact (scientifique ou fixe)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
    
    # 3. Rotation des labels pour plus de clarté
    plt.xticks(rotation=45)
    
    plt.title('Distribution des sommes')
    plt.xlabel('Somme calculée')
    plt.ylabel('Fréquence (Nombre de lignes)')
    
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Ajuste les marges pour que les labels inclinés ne soient pas coupés
    plt.tight_layout()
    
    plt.savefig("hist_sum_Pu.png")
    

def load_and_process_spectra(base_path):
    """
    Load and process all spectrum files from the base directory.
    
    Args:
        base_path (str): Path to directory containing spectra
        
    Returns:
        pd.DataFrame: DataFrame containing all processed spectra
    """
    dataframes = []
    
    # Iterate through all files 
    for filename in os.listdir(f"{base_path}/IDB-all-spectra/"):
        print(f"  Processing file: {filename}")
        file_path = f"{base_path}/IDB-all-spectra/{filename}"
        
        # Parse file
        metadata_dict = parse_spectrum_file(file_path)
        if metadata_dict is None:
            continue
                    
        # Extract spectrum data
        try:

            # Process metadata
            processed_metadata = process_metadata(metadata_dict)
            
            if processed_metadata is not None:
                # Create DataFrame for this file
                df_file = pd.DataFrame.from_dict(processed_metadata)

                dataframes.append(df_file)

        except ValueError as e:
            print(f"Error in file {filename}: {e}")
        
            
    print(len(dataframes))
    # Concatenate all DataFrames
    final_df = pd.concat(dataframes, ignore_index=True, join="inner")

    col = ['238Pu', '239Pu', '240Pu', '241Pu', '242Pu']
    plot_row_sum_histogram(final_df, col)
    
    final_df = final_df.dropna()
    
    return final_df

def clean_and_convert_data(df):
    """
    Clean and convert data to appropriate types.
    
    Args:
        df (pd.DataFrame): DataFrame to clean
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """

    #Removing NAI due to only 6 exemples in the database
    df = df[df["Detector"]!="NaI"]
    
   

    # Convert isotope values to float
    for col in df.columns[2:7]:
        df[col] = df[col].astype(float)

    df['ENER_FIT'] = df['ENER_FIT'].astype(float)

    df = df[df["238Pu"]<20]
    df = df[df["239Pu"]>60]
    df = df[df["240Pu"]<40]

    return df

def filter_and_normalize_data(df):
    """
    Filter data and apply normalization.
    
    Args:
        df (pd.DataFrame): DataFrame to filter and normalize
        
    Returns:
        pd.DataFrame: Filtered and normalized DataFrame
    """
    # Filter: enrichment < 90% and real counting time > 1000s
    # filtered_df = df[
    #     (df["Declared enrichment"] < 90) & 
    #     (df["real counting times"] > 1000)
    # ].copy()
    # filtered_df.reset_index(drop=True, inplace=True)

    filtered_df = df.copy()

    # Select relevant columns
    columns_to_keep = [
        "File name", "Detector", '238Pu', '239Pu', '240Pu', '241Pu', '242Pu', '241Am', 'content', 'ENER_FIT'
    ]
    
    # Check that all columns exist
    available_columns = [col for col in columns_to_keep if col in filtered_df.columns]
    
    final_df = filtered_df[available_columns].copy()

    #Filter outliers
    # final_df = final_df[final_df["Detector"] < 4]
    # final_df = final_df[final_df["Pu-240/Pu"] < 40]

    # # Variable normalization
    # scaler_standard = preprocessing.StandardScaler()
    # scaler_minmax = preprocessing.MinMaxScaler()
    
    # for col in ['real counting times', 'live counting times', "Pu-238/Pu", "Pu-239/Pu", "Pu-240/Pu","Pu-241/Pu", "Pu-242/Pu", "Am-241/Pu"]:
    #     # Normalize counting times (MinMaxScaler on log)
    #     final_df.loc[:, col] = scaler_minmax.fit_transform(
    #         np.log1p(final_df[[col]])
    #     ).flatten()

    
    return final_df

def indices_par_nom(L):
    """Group indices by detector name."""
    d = defaultdict(list)
    for i, nom in enumerate(L):
        d[nom].append(i)
    return dict(d)

def is_composition_valid(composition, bounds, isotope_columns, tolerance=1e-6):
    """
    Check if a composition respects the bounds and sums to ~100%.
    
    Args:
        composition: Array of isotope values
        bounds: Dictionary of (min, max) for each isotope
        isotope_columns: List of isotope names
        tolerance: Tolerance for sum check
        
    Returns:
        bool: True if valid
    """
    # Check sum
    total = composition.sum()
    if abs(total - 100.0) > tolerance:
        return False
    
    # Check bounds
    for i, isotope in enumerate(isotope_columns):
        min_val, max_val = bounds[isotope]
        if composition[i] < min_val or composition[i] > max_val:
            return False
    
    return True

def compute_isotope_bounds(reference_df, isotope_columns, margin=0.0):
    """
    Compute min/max bounds for each isotope from reference data.
    
    Args:
        reference_df: Reference dataframe
        isotope_columns: List of isotope column names
        margin: Safety margin (0.0-0.1) to slightly expand bounds
        
    Returns:
        dict: {isotope_name: (min_val, max_val)}
    """
    bounds = {}
    for isotope in isotope_columns:
        min_val = reference_df[isotope].min()
        max_val = reference_df[isotope].max()
        
        # Add small margin if requested
        if margin > 0:
            range_val = max_val - min_val
            min_val = max(0, min_val - margin * range_val)
            max_val = min(100, max_val + margin * range_val)
        
        print(isotope)
        print(min_val)
        print(max_val)
        bounds[isotope] = (min_val, max_val)
    
    print("\n=== Isotope Bounds ===")
    for isotope, (min_v, max_v) in bounds.items():
        print(f"{isotope}: [{min_v:.4f}, {max_v:.4f}]")
    
    return bounds


def generate_constrained_weights(ref_compositions, selected_indices, bounds, 
                                 isotope_columns, max_attempts=100):
    """
    Generate weights that ensure the resulting composition respects bounds.
    Uses optimization to find valid weights.
    
    Args:
        ref_compositions: All reference compositions
        selected_indices: Indices of selected spectra
        bounds: Dictionary of isotope bounds
        isotope_columns: List of isotope names
        max_attempts: Maximum optimization attempts
        
    Returns:
        np.array: Valid weights or None if failed
    """
    n_combine = len(selected_indices)
    selected_comps = ref_compositions[selected_indices]
    
    # Try random weights first (fast path)
    for attempt in range(10):
        weights = np.random.dirichlet(0.2*np.ones(n_combine))
        combined = np.sum(weights[:, np.newaxis] * selected_comps, axis=0)
        combined_norm = (combined / combined.sum()) * 100.0
        
        if is_composition_valid(combined_norm, bounds, isotope_columns):
            return weights
    
    # If random fails, use optimization
    def objective(w):
        """Minimize deviation from uniform weights while respecting constraints."""
        return np.sum((w - 1.0/n_combine)**2)
    
    def constraint_sum(w):
        """Weights must sum to 1."""
        return np.sum(w) - 1.0
    
    def constraint_bounds_factory(isotope_idx):
        """Create constraint functions for each isotope."""
        min_val, max_val = bounds[isotope_columns[isotope_idx]]
        
        def constraint_min(w):
            combined = np.sum(w * selected_comps[:, isotope_idx])
            total = np.sum(w[:, np.newaxis] * selected_comps, axis=0).sum()
            normalized = (combined / total) * 100.0
            return normalized - min_val
        
        def constraint_max(w):
            combined = np.sum(w * selected_comps[:, isotope_idx])
            total = np.sum(w[:, np.newaxis] * selected_comps, axis=0).sum()
            normalized = (combined / total) * 100.0
            return max_val - normalized
        
        return constraint_min, constraint_max
    
    # Build constraints
    constraints = [{'type': 'eq', 'fun': constraint_sum}]
    
    for i in range(len(isotope_columns)):
        c_min, c_max = constraint_bounds_factory(i)
        constraints.append({'type': 'ineq', 'fun': c_min})
        constraints.append({'type': 'ineq', 'fun': c_max})
    
    # Initial guess: uniform weights
    w0 = np.ones(n_combine) / n_combine
    
    # Bounds for weights: [0, 1]
    weight_bounds = [(0.01, 0.99) for _ in range(n_combine)]
    
    # Optimize
    result = minimize(
        objective, w0, 
        method='SLSQP',
        bounds=weight_bounds,
        constraints=constraints,
        options={'maxiter': 200, 'ftol': 1e-8}
    )
    
    if result.success:
        weights = result.x
        weights = weights / weights.sum()  # Renormalize
        return weights
    
    return None


def generate_linear_combination_spectra_constrained(
    reference_df, 
    n_synthetic=1000,
    min_spectra=2, 
    max_spectra=4,
    random_seed=42, 
    add_noise=True,
    use_density_sampling=True,
    isotope_margin=0.0,
    max_retries=50,
):
    """
    Generate synthetic spectra by constrained linear combination.
    Ensures isotope compositions stay within observed min/max bounds.
    
    Args:
        reference_df: Reference dataset
        n_synthetic: Number of synthetic samples
        min_spectra: Minimum spectra to combine
        max_spectra: Maximum spectra to combine
        random_seed: Random seed
        add_noise: Add Poisson noise
        use_density_sampling: Favor under-represented regions
        isotope_margin: Safety margin for bounds (0.0-0.1)
        max_retries: Max attempts per synthetic sample
        
    Returns:
        pd.DataFrame: Synthetic dataset
    """
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    print(f"\n=== Generating {n_synthetic} constrained synthetic samples ===")
    print(f"Combining {min_spectra} to {max_spectra} spectra per sample")
    print(f"Isotope margin: {isotope_margin}")
    
   
    isotope_columns = ['238Pu', '239Pu', '240Pu', '241Pu', '242Pu',]
    
    #Select only good spectrum
    clean_reference_df = reference_df[(reference_df[isotope_columns] != -100).all(axis=1)]


    sum_ = clean_reference_df[isotope_columns].sum(axis=1)
    nb_valides = sum_.between(99, 101).sum()
    print(nb_valides/len(clean_reference_df))

    
    # Compute isotope bounds
    isotope_bounds = compute_isotope_bounds(clean_reference_df, isotope_columns, isotope_margin)
    
    # Extract reference data
    ref_spectra = np.array([spec for spec in clean_reference_df["content"].values])
    ref_compositions = clean_reference_df[isotope_columns].values
    ref_compositions = ref_compositions.astype(float)

    ref_compositions_all = clean_reference_df[isotope_columns].values
    ref_compositions_all = ref_compositions_all.astype(float)

    n_references = len(clean_reference_df)
    
    # Detector information
    detectors = np.array([spec for spec in clean_reference_df["Detector"].values])
    print(clean_reference_df["Detector"].value_counts())
    name_detector = np.unique(detectors)
    detector_dict = indices_par_nom(detectors)
    detector_quanti_dict = dict(zip(clean_reference_df["Detector"], clean_reference_df["Detector quanti"]))
    
    # Density-based sampling setup
    detector_dict_common_indices = {}
    detector_dict_rare_indices = {}
    detector_dict_inv_dens = {}
    
    if use_density_sampling:
        print("\n=== Computing density for sampling strategy ===")
        iso_values = clean_reference_df[isotope_columns].values.T
        iso_values = iso_values.astype(float)
        kde_iso = gaussian_kde(iso_values)
        dens = kde_iso(iso_values)
        
        for name in name_detector:
            mask = np.array(detector_dict[name])
            low_density_mask = dens[mask] < np.percentile(dens[mask], 40)
            rare_indices = np.where(low_density_mask)[0]
            common_indices = np.where(~low_density_mask)[0]
            
            detector_dict_common_indices[name] = mask[common_indices]
            detector_dict_rare_indices[name] = mask[rare_indices]
            
            inv_dens = 1 / (dens[mask] + 1e-6)
            inv_dens /= inv_dens.sum()
            detector_dict_inv_dens[name] = inv_dens
    
    # Generation statistics
    synthetic_data = []
    failed_attempts = 0
    total_attempts = 0
    
    sample_idx = 0
    while sample_idx < n_synthetic:
        # Select detector
        n_detector = np.random.randint(0, len(name_detector))
        detector_name = name_detector[n_detector]
        
        # Select number of spectra to combine
        n_combine = np.random.randint(min_spectra, max_spectra + 1)
        
        # Try to generate valid sample
        success = False
        for retry in range(max_retries):
            total_attempts += 1
            
            # Select indices based on density sampling
            if use_density_sampling:
                if np.random.rand() < 0.5 and len(detector_dict_rare_indices[detector_name]) > 0:
                    # Include rare spectrum
                    rare_idx = np.random.choice(detector_dict_rare_indices[detector_name])
                    if n_combine > 1 and len(detector_dict_common_indices[detector_name]) >= n_combine - 1:
                        other_indices = np.random.choice(
                            detector_dict_common_indices[detector_name], 
                            size=n_combine - 1, 
                            replace=False
                        )
                        selected_indices = np.concatenate([[rare_idx], other_indices])
                    else:
                        
                        selected_indices = np.random.choice(
                            detector_dict[detector_name],
                            size=n_combine,
                            replace=False
                        )
                else:
                    # Weighted sampling by inverse density
                    available_indices = detector_dict[detector_name]
                    if len(available_indices) >= n_combine:
                        selected_indices = np.random.choice(
                            available_indices,
                            size=n_combine,
                            replace=False,
                            p=detector_dict_inv_dens[detector_name]
                        )
                    else:
                        continue
            else:
                # Simple uniform sampling
                available_indices = detector_dict[detector_name]
                if len(available_indices) >= n_combine:
                    selected_indices = np.random.choice(
                        available_indices,
                        size=n_combine,
                        replace=False
                    )
                else:
                    continue
            
            # Generate constrained weights
            weights = generate_constrained_weights(
                ref_compositions, 
                selected_indices, 
                isotope_bounds,
                isotope_columns
            )
            
            if weights is None:
                failed_attempts += 1
                continue
            
            # Combine spectra
            combined_spectrum = np.zeros_like(ref_spectra[0], dtype=float)
            for idx, weight in zip(selected_indices, weights):
                combined_spectrum += weight * ref_spectra[idx]
            
            # Add Poisson noise
            if add_noise:
                combined_spectrum = np.maximum(combined_spectrum, 0)
                combined_spectrum = np.random.poisson(combined_spectrum)
            
            combined_spectrum = combined_spectrum.astype(int)
            
            # --- Calcul de la composition combinée ---
            combined_composition = np.zeros(len(isotope_columns), dtype=float)
            for idx, weight in zip(selected_indices, weights):
                # On utilise ref_compositions_all qui contient les 9 isotopes
                combined_composition += weight * ref_compositions[idx].astype(float)
            
            # Normalisation globale
            composition_sum = combined_composition.sum()

            combined_composition = np.zeros(len(isotope_columns), dtype=float)
            for idx, weight in zip(selected_indices, weights):
                # On utilise ref_compositions_all qui contient les 9 isotopes
                combined_composition += weight * ref_compositions_all[idx].astype(float)

            if composition_sum > 0:
                combined_composition = (combined_composition / composition_sum) * 100.0
            ref_row = clean_reference_df.iloc[selected_indices[0]]

            # --- Traitement spécifique U ou Pu ---
            
            # On vérifie la validité du bloc Uranium (index 0 à 3)
           
            sum_Pu = combined_composition.sum()
            if not (99 <= sum_Pu <= 101):
                print("invalid Pu")

            synthetic_row = {
                'File name': f'synthetic_constrained_{sample_idx:04d}.txt',
                'Detector': detector_name,
                'Detector quanti': detector_quanti_dict[detector_name],
                'content': combined_spectrum,
                'ENER_FIT': ref_row['ENER_FIT'],
                '238Pu': combined_composition[0], 
                '239Pu': combined_composition[1],
                '240Pu': combined_composition[2], 
                '241Pu': combined_composition[3],
                '242Pu': combined_composition[4],
                'n_combined_spectra': n_combine,
                'combination_weights': weights.tolist(),
                'combined_indices': selected_indices.tolist()
            }
            synthetic_data.append(synthetic_row)
            sample_idx += 1
            success = True
            break
        
        if not success:
            print(f"Warning: Failed to generate sample {sample_idx} after {max_retries} retries")
            sample_idx += 1  # Skip this sample
        
        if (sample_idx) % 100 == 0 and sample_idx > 0:
            print(f"  Generated {sample_idx}/{n_synthetic} samples (failed: {failed_attempts}/{total_attempts})")
    
    synthetic_df = pd.DataFrame(synthetic_data)
    
    # Validation
    print(f"\n=== Validation ===")
    composition_sums = synthetic_df[isotope_columns].sum(axis=1)
    print(f"Composition sum - Mean: {composition_sums.mean():.6f}%, Std: {composition_sums.std():.6e}%")
    
    print(f"\nIsotope bounds validation:")
    all_valid = True
    for isotope in isotope_columns:
        min_bound, max_bound = isotope_bounds[isotope]
        actual_min = synthetic_df[isotope].min()
        actual_max = synthetic_df[isotope].max()
        
        valid = (actual_min >= min_bound) and (actual_max <= max_bound)
        status = "✓" if valid else "✗"
        
        print(f"  {isotope}: [{actual_min:.4f}, {actual_max:.4f}] vs [{min_bound:.4f}, {max_bound:.4f}] {status}")
        all_valid = all_valid and valid
    
    print(f"\nAll bounds respected: {'✓ YES' if all_valid else '✗ NO'}")
    print(f"Success rate: {len(synthetic_data)}/{total_attempts} ({100*len(synthetic_data)/total_attempts:.1f}%)")
    
    return synthetic_df


def create_histograms(df, save_path="./only_Pu2/", normalized = False):
    """
    Create histograms to visualize variable distributions.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        save_path (str): Path where to save histograms
    """
    # Columns to exclude from histograms
    exclude_columns = ['File name', 'Detector', 'content', 'combination_weights', 'combined_indices']
    
    for column in df.columns:
        if normalized:
            fig_title = f"{save_path}hist_{column.replace(' ', '_').replace('/', '_')}_normalized.png"
        else:
            fig_title = f"{save_path}hist_{column.replace(' ', '_').replace('/', '_')}.png"

        if column not in exclude_columns:
            print(column)
            plt.figure(figsize=(10, 6))
            plt.hist(df[df[column] >= 0][[column]], bins=30, edgecolor="black", alpha=0.7, color='skyblue')
            plt.xlabel(column)
            plt.ylabel("Frequency")
            plt.title(f"Distribution of {column}")
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.savefig(fig_title, dpi=300)
            plt.close()

        if column not in exclude_columns:
            print(column)
            plt.figure(figsize=(10, 6))
            plt.hist(df[column] >= 0, bins=[-0.5, 0.5, 1.5], edgecolor="black", alpha=0.7, color='skyblue')
            plt.xticks([0,1], ['Unkonwn', 'Known'])
            plt.ylabel('Count')
            plt.title(f"Proportion known for {column}")
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.savefig(f"{save_path}hist_{column.replace(' ', '_').replace('/', '_')}_is_known.png", dpi=300)
            plt.close()

        if column == "Detector":
            plt.figure(figsize=(10, 6))
            df[column].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
            plt.xlabel(column)
            plt.ylabel("Frequency")
            plt.title(f"Distribution of {column}")
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.savefig(f"{save_path}hist_{column.replace(' ', '_').replace('/', '_')}.png", dpi=300)
            plt.close()
    

def stratified_train_val_split(df, train_ratio=0.8, random_seed=4):
    """
    Perform stratified split based on detector type.
    
    Args:
        df (pd.DataFrame): Input dataframe
        train_ratio (float): Ratio of data to use for training
        random_seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_dataset, val_dataset)
    """
    random.seed(random_seed)
    
    train_dataset = pd.DataFrame(columns=df.columns.tolist())
    val_dataset = pd.DataFrame()


    df["Detector quanti"] = df["Detector"].astype("category").cat.codes

    # Get unique detector types
    detector_types = df['Detector quanti'].value_counts().index
    print("Detector type distribution:")
    print(df['Detector quanti'].value_counts())
    
    # Iterate through each detector type
    for detector_type in detector_types:
        detector_subset = df[df['Detector quanti'] == detector_type]
        print(f"Samples for detector type {detector_type}: {len(detector_subset)}")
        
        # Create shuffled indices for this subset
        indices = list(range(len(detector_subset)))
        random.shuffle(indices)
        
        # Split indices into train and validation
        split_point = int(np.floor(train_ratio * len(detector_subset)))
        print(f"Train samples: {split_point}, Val samples: {len(detector_subset) - split_point}")
        
        train_indices = indices[:split_point]
        val_indices = indices[split_point:]
        
        # Add samples to respective datasets
        train_dataset = pd.concat([train_dataset, detector_subset.iloc[train_indices]], ignore_index=True)
        val_dataset = pd.concat([val_dataset, detector_subset.iloc[val_indices]], ignore_index=True)
    
    return train_dataset, val_dataset



def normalize_spectra(train_df, val_df, test_df, save_path="./only_Pu2/"):
    """
    Normalize spectra by applying log(1+x) then normalization by max amplitude.
    Store global parameters efficiently in a separate metadata file.
    
    Args:
        train_df, val_df, test_df (pd.DataFrame): DataFrames of different sets
        save_path (str): Path to save metadata file
        
    Returns:
        tuple: DataFrames with normalized spectra and normalization config
    """

    # Extract spectra
    train_spectra = np.array([spectrum for spectrum in train_df["content"].values])
    val_spectra = np.array([spectrum for spectrum in val_df["content"].values])
    test_spectra = np.array([spectrum for spectrum in test_df["content"].values])
    
    print(np.min(train_spectra), np.min(val_spectra), np.min(test_spectra))
    # Apply log(1+x)
    train_log_spectra = np.log1p(train_spectra)
    val_log_spectra = np.log1p(val_spectra)
    test_log_spectra = np.log1p(test_spectra)
    
    # Calculate maximum amplitudes
    train_max_amp = np.max(train_log_spectra, axis=1)[:, np.newaxis]
    val_max_amp = np.max(val_log_spectra, axis=1)[:, np.newaxis]
    test_max_amp = np.max(test_log_spectra, axis=1)[:, np.newaxis]
    
    # Calculate global bounds for normalization
    global_max = max(np.max(train_max_amp), np.max(val_max_amp), np.max(test_max_amp))
    global_min = min(np.min(train_max_amp), np.min(val_max_amp), np.min(test_max_amp))
    
    # Normalize spectra by their maximum amplitude
    train_normalized = train_log_spectra / train_max_amp
    val_normalized = val_log_spectra / val_max_amp
    test_normalized = test_log_spectra / test_max_amp
    
    # Min-max normalization of maximum amplitudes
    train_max_scaled = (train_max_amp - global_min) / (global_max - global_min)
    val_max_scaled = (val_max_amp - global_min) / (global_max - global_min)
    test_max_scaled = (test_max_amp - global_min) / (global_max - global_min)


    # scaler_minmax = preprocessing.MinMaxScaler()
    
    
    
    # Create normalization metadata
    normalization_metadata = {
        "global_min": float(global_min),
        "global_max": float(global_max),
        "energy": np.arange(0, 4096).tolist(),  # 4096 channels for this data format
        "normalization_method": "log1p_then_max_amplitude_scaling",
        "timestamp": pd.Timestamp.now().isoformat(),
        "dataset_info": {
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
            "total_samples": len(train_df) + len(val_df) + len(test_df)
        },
        "processing_steps": [
            "1. Apply log(1+x) transformation to spectra",
            "2. Calculate max amplitude per spectrum",
            "3. Normalize each spectrum by its max amplitude", 
            "4. Scale max amplitudes using global min-max normalization"
        ]
    }
    
    # Save metadata to file
    metadata_path = f"{save_path}/normalization_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(normalization_metadata, f, indent=2)
    print(f"Normalization metadata saved to: {metadata_path}")
    
    # Update DataFrames (WITHOUT storing redundant global values)
    datasets = [
        (train_df.copy(), train_normalized, train_max_amp, train_max_scaled),
        (val_df.copy(), val_normalized, val_max_amp, val_max_scaled),
        (test_df.copy(), test_normalized, test_max_amp, test_max_scaled)
    ]
    
    normalized_datasets = []
    for df, normalized_spectra, max_amp, max_scaled in datasets:
        df["content"] = [spectrum.tolist() for spectrum in normalized_spectra]
        df["max_amplitude"] = max_amp.flatten()  # Individual max amplitude
        df["max_amplitude_scaled"] = max_scaled.flatten()  # Scaled individual max
        # DO NOT store global_min/global_max in each row - use metadata file instead
        print(df.columns)
        for col in ['238Pu', '239Pu', '240Pu', '241Pu', '242Pu']:
            # Normalize counting times (MinMaxScaler on log)
            # df[col] = scaler_minmax.fit_transform(
            #     df[[col]]
            # ).flatten()
            df[col] = df[col] / 100.0
        # Optional: Store reference to metadata file in DataFrame attributes
        df.attrs['normalization_metadata_file'] = metadata_path
        df.attrs['global_min'] = float(global_min)
        df.attrs['global_max'] = float(global_max)
        
        normalized_datasets.append(df)
    
    return tuple(normalized_datasets), normalization_metadata

def save_datasets(datasets, filenames):
    """
    Save datasets in JSON format.
    
    Args:
        datasets (tuple): Tuple containing DataFrames to save
        filenames (list): List of filenames
    """
    for dataset, filename in zip(datasets, filenames):
        dataset.to_json(filename)
        print(dataset.columns)
        print(f"Dataset saved: {filename}")

def main():
    """
    Main function orchestrating the entire processing workflow.
    """
    # Configuration
    base_path = os.getcwd()
    print(f"Working directory: {base_path}")
    
    # =============================================================================
    # 1. DATA LOADING AND INITIAL PROCESSING
    # =============================================================================
    
    print("\n=== Loading spectra ===")
    df_raw = load_and_process_spectra(base_path)
    
    # 2. Data cleaning and conversion
    print("\n=== Data cleaning ===")
    df_clean = clean_and_convert_data(df_raw)
    
    # 3. Save raw data
    df_clean.to_json("only_Pu2/dataset_before_normalization.json")
    create_histograms(df_clean)
    
    # 4. Filtering and normalization
    print("\n=== Filtering and normalization ===")
    df_filtered = filter_and_normalize_data(df_clean)
    
    # 5. Save filtered data
    df_filtered.to_json("only_Pu2/dataset_after_normalization.json")
    

    # 6. Create histograms
    print("\n=== Creating histograms ===")

    # =============================================================================
    # 2. DATASET SPLITTING
    # =============================================================================
    
    print("\nPerforming stratified train/validation split...")
    
    # Perform stratified split based on detector type
    train_dataset, val_dataset = stratified_train_val_split(df_filtered, train_ratio=0.8, random_seed=4)
    train_dataset.reset_index(inplace=True, drop=True)
    
    # Further split validation set into validation and test sets
    print("Splitting validation set into validation and test sets...")
    val_indices = np.arange(len(val_dataset))
    np.random.shuffle(val_indices)
    
    # Split validation set in half
    split_point = len(val_dataset) // 2
    val_ixs, test_ixs = val_indices[:split_point], val_indices[split_point:]
    
    test_dataset = val_dataset.iloc[test_ixs].reset_index(drop=True)
    val_dataset = val_dataset.iloc[val_ixs].reset_index(drop=True)
    


    augmented_dataset1 = generate_linear_combination_spectra_constrained(train_dataset, n_synthetic=2500, 
                                    min_spectra=2, max_spectra=10,
                                    random_seed=42, add_noise=True)
    
    augmented_dataset2 = generate_linear_combination_spectra_constrained(train_dataset, n_synthetic=2500, 
                                    min_spectra=2, max_spectra=4,
                                    random_seed=42, add_noise=True)
    
    augmented_dataset = pd.concat([augmented_dataset1, augmented_dataset2], ignore_index = True)
    #Save unnormalized datasets
    save_datasets(
        (augmented_dataset, val_dataset, test_dataset),
        ["only_Pu2/train_before_normalization_reduced_augmented.json", 
         "only_Pu2/val_before_normalization_reduced.json", 
         "only_Pu2/test_before_normalization_reduced.json"]
    )

    
    # =============================================================================
    # 3. DATA NORMALIZATION
    # =============================================================================
    
    print("\nApplying normalization to spectral data...")
    
    # Apply normalization
    (train_norm, val_norm, test_norm), norm_metadata = normalize_spectra(
        augmented_dataset, val_dataset, test_dataset, save_path="./only_Pu2/"
    )
    
    create_histograms(train_norm, save_path = "./only_Pu2/augmented_")
 
    # =============================================================================
    # 4. FINAL DATASET PREPARATION
    # =============================================================================
    
    # Final save
    print("\n=== Final save ===")
    save_datasets(
        (train_norm, val_norm, test_norm),
        ["only_Pu2/train.json", 
         "only_Pu2/val.json", 
         "only_Pu2/test.json"]
    )
    
    print(f"Global normalization bounds: min={norm_metadata['global_min']:.6f}, max={norm_metadata['global_max']:.6f}")
    print("\n=== Processing completed successfully ===")

if __name__ == "__main__":
    main()