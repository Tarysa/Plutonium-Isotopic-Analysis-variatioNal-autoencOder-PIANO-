import pandas as pd
import os 
import numpy as np
import re
import sys
import matplotlib.pyplot as plt
import random
import json
from sklearn import preprocessing
from collections import defaultdict
from scipy.stats import gaussian_kde
from scipy.optimize import minimize

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

    # Extract metadata (first 10 lines)
    metadata_lines = [file_lines[i].strip() for i in range(10) if i != 2]
    
    # Parse metadata
    parsed_metadata = []
    for line in metadata_lines:
        parts = line.split(":")
        for part in parts:
            parsed_metadata.extend(part.split('\t'))
    
    # Clean and create metadata dictionary
    clean_metadata = [item for item in parsed_metadata if len(item) > 1 and item != ' ']
    
    metadata_dict = {}
    for i in range(len(clean_metadata) // 2):
        key = re.sub(r"\s+", " ", clean_metadata[2*i]).strip()
        # Normalize units
        key = key.replace("(cm )", "(cm)").replace("( cm )", "(cm)").replace("( cm)", "(cm)")
        key = key.replace(" (s)", "(s)").replace("(s) ", "(s)").replace("( mm)", "(mm)")
        
        value = re.sub(r"\s+", " ", clean_metadata[2*i + 1]).strip()
        value = re.sub(r'^([+-]?\d+(?:\.\d+)?)(?:\s+[+-]?\d+(?:\.\d+)?)$', r'\1', value)

        metadata_dict[key] = value

    return metadata_dict, file_lines

def extract_spectrum_data(file_lines):
    """
    Extract spectrum data (from line 8 onwards).
    
    Args:
        file_lines (list): List of file lines
        
    Returns:
        np.array: Array containing spectrum data
    """
    content = []
    for line in file_lines[10:]:
        values = line.strip().split()
        content.extend([int(val) for val in values if val])
    
    content_array = np.array(content)
    
    # Verify that spectrum contains 8192 channels
    if len(content_array) != 8192:
        raise ValueError(f"Spectrum must contain 8192 channels, found: {len(content_array)}")
    
    return content_array

def process_metadata(metadata_dict, detector_count):
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
        "File name", "Detector", "live counting times", 
        "real counting times", "Detector quanti", "FWHM at 208 keV (keV)", "Pu-238/Pu", 
        "Pu-239/Pu", "Pu-240/Pu","Pu-241/Pu", "Pu-242/Pu", "Am-241/Pu"
    ]
    
    # Separate live/real counting times
    if "Live/real counting times(s)" in metadata_dict:
        live_time, real_time = metadata_dict["Live/real counting times(s)"].split("/")
        metadata_dict["live counting times"] = live_time
        metadata_dict["real counting times"] = real_time
        
    # Add detector identifier
    metadata_dict["Detector quanti"] = detector_count
    
    # Create final dictionary with only necessary columns
    processed_dict = {}
    for col in expected_columns:
        if col in metadata_dict:
            processed_dict[col] = metadata_dict[col]
        else:
            print(f"Missing column: {col}")
            print(f"Available columns: {list(metadata_dict.keys())}")
            sys.exit(f"Fatal error: missing information {col}")
    
    return processed_dict

def load_and_process_spectra(base_path):
    """
    Load and process all spectrum files from the base directory.
    
    Args:
        base_path (str): Path to directory containing spectra
        
    Returns:
        pd.DataFrame: DataFrame containing all processed spectra
    """
    dataframes = []
    
    # Iterate through all detectors
    for detector_count, detector_type in enumerate(os.listdir(f"{base_path}/Pu")):
        print(f"Processing detector: {detector_type}")
        
        # Iterate through all files for this detector
        for filename in os.listdir(f"{base_path}/Pu/{detector_type}"):
            print(f"  Processing file: {filename}")
            file_path = f"{base_path}/Pu/{detector_type}/{filename}"
            
            # Parse file
            result = parse_spectrum_file(file_path)
            if result is None:
                continue
                
            metadata_dict, file_lines = result
            
            # Extract spectrum data
            try:
                spectrum_data = extract_spectrum_data(file_lines)

                # Process metadata
                processed_metadata = process_metadata(metadata_dict, detector_count)
                processed_metadata["content"] = [spectrum_data]
                
                # Create DataFrame for this file
                df_file = pd.DataFrame.from_dict(processed_metadata)
                dataframes.append(df_file)

            except ValueError as e:
                print(f"Error in file {filename}: {e}")
            
            
    
    # Concatenate all DataFrames
    final_df = pd.concat(dataframes, ignore_index=True, join="inner")
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
    print(df.columns)

    # Convert counting times to integers
    df.iloc[:, 2] = [int(val.split(" ")[0]) for val in df.iloc[:, 2]]
    df.iloc[:, 3] = [int(val.split(" ")[0]) for val in df.iloc[:, 3]]
    
    # Convert FWHM to float
    df.iloc[:, 5] = [float(val.split(" ")[0]) for val in df.iloc[:, 5]]

    # Convert isotope values to float
    for i in range(6, 12):
        df.iloc[:, i] = [float(val.split(" ")[0]) for val in df.iloc[:, i]]

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
        'File name', 'Detector', 'Detector quanti', 
        'content', 'real counting times', 'live counting times', 
        "Pu-238/Pu", "Pu-239/Pu", "Pu-240/Pu","Pu-241/Pu", "Pu-242/Pu", "Am-241/Pu", "FWHM at 208 keV (keV)",
    ]
    
    # Check that all columns exist
    available_columns = [col for col in columns_to_keep if col in filtered_df.columns]
    
    final_df = filtered_df[available_columns].copy()

    #Filter outliers
    final_df = final_df[final_df["Pu-239/Pu"] > 60]
    final_df = final_df[final_df["Pu-240/Pu"] < 40]
    print(final_df.columns)

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
    max_spectra=10,
    random_seed=42, 
    add_noise=True,
    use_density_sampling=True,
    isotope_margin=0.0,
    max_retries=50
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
    
    isotope_columns = ['Pu-238/Pu', 'Pu-239/Pu', 'Pu-240/Pu', 'Pu-241/Pu', 'Pu-242/Pu']
    
    # Compute isotope bounds
    isotope_bounds = compute_isotope_bounds(reference_df, isotope_columns, isotope_margin)
    
    # Extract reference data
    ref_spectra = np.array([spec for spec in reference_df["content"].values])
    ref_compositions = reference_df[isotope_columns].values
    n_references = len(reference_df)
    
    # Detector information
    detectors = np.array([spec for spec in reference_df["Detector"].values])
    name_detector = np.unique(detectors)
    detector_dict = indices_par_nom(detectors)
    detector_quanti_dict = dict(zip(reference_df["Detector"], reference_df["Detector quanti"]))
    
    # Density-based sampling setup
    detector_dict_common_indices = {}
    detector_dict_rare_indices = {}
    detector_dict_inv_dens = {}
    
    if use_density_sampling:
        print("\n=== Computing density for sampling strategy ===")
        iso_values = reference_df[isotope_columns].values.T
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
            
            # Calculate combined composition
            combined_composition = np.zeros(len(isotope_columns))
            for idx, weight in zip(selected_indices, weights):
                combined_composition += weight * ref_compositions[idx]
            
            # Normalize
            composition_sum = combined_composition.sum()
            if composition_sum > 0:
                combined_composition = (combined_composition / composition_sum) * 100.0
            
            # Verify bounds (should always pass)
            if not is_composition_valid(combined_composition, isotope_bounds, isotope_columns):
                failed_attempts += 1
                continue
            
            # Success! Create synthetic row
            ref_row = reference_df.iloc[selected_indices[0]]
            
            synthetic_row = {
                'File name': f'synthetic_constrained_{sample_idx:04d}.txt',
                'Detector': detector_name,
                'Detector quanti': detector_quanti_dict[detector_name],
                'content': combined_spectrum,
                'real counting times': ref_row['real counting times'],
                'live counting times': ref_row['live counting times'],
                'FWHM at 208 keV (keV)': ref_row['FWHM at 208 keV (keV)'],
                'Pu-238/Pu': combined_composition[0],
                'Pu-239/Pu': combined_composition[1],
                'Pu-240/Pu': combined_composition[2],
                'Pu-241/Pu': combined_composition[3],
                'Pu-242/Pu': combined_composition[4],
                'Am-241/Pu': ref_row['Am-241/Pu'],
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


def create_histograms(df, save_path="./", normalized = False):
    """
    Create histograms to visualize variable distributions.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data
        save_path (str): Path where to save histograms
    """
    # Columns to exclude from histograms
    exclude_columns = ['File name', 'content', 'combination_weights', 'combined_indices']
    
    for column in df.columns:
        if normalized:
            fig_title = f"{save_path}hist_{column.replace(' ', '_').replace('/', '_')}_normalized.png"
        else:
            fig_title = f"{save_path}hist_{column.replace(' ', '_').replace('/', '_')}.png"

        if column not in exclude_columns:
            print(column)
            plt.figure(figsize=(10, 6))
            plt.hist(df[column], bins=30, edgecolor="black", alpha=0.7, color='skyblue')
            plt.xlabel(column)
            plt.ylabel("Frequency")
            plt.title(f"Distribution of {column}")
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.savefig(fig_title, dpi=300)
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


def normalize_spectra(train_df, val_df, test_df, save_path="./"):
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
        
        for col in ["Pu-238/Pu", "Pu-239/Pu", "Pu-240/Pu","Pu-241/Pu", "Pu-242/Pu"]:
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
    df_clean.to_json("dataset_before_normalization.json")
    create_histograms(df_clean)
    
    # 4. Filtering and normalization
    print("\n=== Filtering and normalization ===")
    df_filtered = filter_and_normalize_data(df_clean)
    
    # 5. Save filtered data
    df_filtered.to_json("dataset_after_normalization.json")
    
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
    
    # augmented_dataset = create_synthetic_dataset(train_dataset, n_synthetic=1000, random_seed=42, 
    #                         interpolation_method='weighted_average')

    # augmented_dataset = generate_linear_combination_spectra_constrained(train_dataset, n_synthetic=5000, 
    #                                    min_spectra=2, max_spectra=10,
    #                                    random_seed=42, add_noise=True)

    augmented_dataset = generate_linear_combination_spectra_constrained(train_dataset, n_synthetic=5000, 
                                    min_spectra=2, max_spectra=10,
                                    random_seed=42, add_noise=True)

    
    # Save unnormalized datasets
    # save_datasets(
    #     (train_dataset, val_dataset, test_dataset),
    #     ["train_before_normalization_reduced.json", 
    #      "val_before_normalization_reduced.json", 
    #      "test_before_normalization_reduced.json"]
    # )

    
    save_datasets(
        (augmented_dataset, val_dataset, test_dataset),
        ["train_before_normalization_reduced_augmented.json", 
         "val_before_normalization_reduced.json", 
         "test_before_normalization_reduced.json"]
    )
    
    # =============================================================================
    # 3. DATA NORMALIZATION
    # =============================================================================
    
    print("\nApplying normalization to spectral data...")
    
    # Apply normalization
    (train_norm, val_norm, test_norm), norm_metadata = normalize_spectra(
        augmented_dataset, val_dataset, test_dataset, save_path="./"
    )
    
    create_histograms(train_norm, save_path = "./augmented_")
 
    # =============================================================================
    # 4. FINAL DATASET PREPARATION
    # =============================================================================
    
    # Final save
    print("\n=== Final save ===")
    save_datasets(
        (train_norm, val_norm, test_norm),
        ["train.json", 
         "val.json", 
         "test.json"]
    )
    
    print(f"Global normalization bounds: min={norm_metadata['global_min']:.6f}, max={norm_metadata['global_max']:.6f}")
    print("\n=== Processing completed successfully ===")

if __name__ == "__main__":
    main()