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

# ==========================================
# 1. PARSING ET EXTRACTION DES DONNÉES
# ==========================================

def parse_spectrum_file(file_path):
    """Parse un fichier de spectre gamma et extrait les métadonnées."""
    try:
        with open(file_path, encoding="utf8", errors='ignore') as f:
            file_lines = f.readlines()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

    metadata_lines = [file_lines[i].strip() for i in range(10) if i != 2]
    parsed_metadata = []
    for line in metadata_lines:
        parts = line.split(":")
        for part in parts:
            parsed_metadata.extend(part.split('\t'))
    
    clean_metadata = [item for item in parsed_metadata if len(item) > 1 and item != ' ']
    metadata_dict = {}
    for i in range(len(clean_metadata) // 2):
        key = re.sub(r"\s+", " ", clean_metadata[2*i]).strip()
        key = key.replace("(cm )", "(cm)").replace("( cm )", "(cm)").replace("( cm)", "(cm)")
        key = key.replace(" (s)", "(s)").replace("(s) ", "(s)").replace("( mm)", "(mm)")
        value = re.sub(r"\s+", " ", clean_metadata[2*i + 1]).strip()
        value = re.sub(r'^([+-]?\d+(?:\.\d+)?)(?:\s+[+-]?\d+(?:\.\d+)?)$', r'\1', value)
        metadata_dict[key] = value

    return metadata_dict, file_lines

def extract_spectrum_data(file_lines):
    """Extrait les données brutes du spectre (canaux)."""
    content = []
    for line in file_lines[10:]:
        values = line.strip().split()
        content.extend([int(val) for val in values if val])
    
    content_array = np.array(content)
    if len(content_array) != 8192:
        raise ValueError(f"Spectrum must contain 8192 channels, found: {len(content_array)}")
    return content_array

# ==========================================
# 2. PRÉ-TRAITEMENT ET NETTOYAGE
# ==========================================

def process_metadata(metadata_dict, detector_count):
    """Normalise les colonnes de métadonnées."""
    expected_columns = [
        "File name", "Detector", "live counting times", 
        "real counting times", "Detector quanti", "FWHM at 208 keV (keV)", "Pu-238/Pu", 
        "Pu-239/Pu", "Pu-240/Pu","Pu-241/Pu", "Pu-242/Pu", "Am-241/Pu"
    ]
    
    if "Live/real counting times(s)" in metadata_dict:
        live_time, real_time = metadata_dict["Live/real counting times(s)"].split("/")
        metadata_dict["live counting times"] = live_time
        metadata_dict["real counting times"] = real_time
        
    metadata_dict["Detector quanti"] = detector_count
    processed_dict = {col: metadata_dict.get(col, "0") for col in expected_columns}
    return processed_dict

def clean_and_convert_data(df):
    """Convertit les types de colonnes (str -> float/int)."""
    df = df.copy()
    df.iloc[:, 2] = [int(str(val).split(" ")[0]) for val in df.iloc[:, 2]]
    df.iloc[:, 3] = [int(str(val).split(" ")[0]) for val in df.iloc[:, 3]]
    df.iloc[:, 5] = [float(str(val).split(" ")[0]) for val in df.iloc[:, 5]]
    for i in range(6, 12):
        df.iloc[:, i] = [float(str(val).split(" ")[0]) for val in df.iloc[:, i]]
    return df

def filter_and_normalize_data(df):
    """Filtre les outliers et sélectionne les colonnes utiles."""
    columns_to_keep = [
        'File name', 'Detector', 'Detector quanti', 'content', 
        'real counting times', 'live counting times', "Pu-238/Pu", 
        "Pu-239/Pu", "Pu-240/Pu","Pu-241/Pu", "Pu-242/Pu", "Am-241/Pu", "FWHM at 208 keV (keV)",
    ]
    final_df = df[columns_to_keep].copy()
    # Filtres spécifiques
    final_df = final_df[(final_df["Pu-239/Pu"] > 60) & (final_df["Pu-240/Pu"] < 40)]
    return final_df

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

# ==========================================
# 3. GÉNÉRATION SYNTHÉTIQUE (VERSION DIRICHLET)
# ==========================================

def generate_linear_combination_spectra_constrained(
    reference_df, n_synthetic=1000, min_spectra=2, max_spectra=10, 
    random_seed=42, add_noise=True, alpha=0.5
):
    """
    Génère de nouveaux spectres par mélange linéaire.
    Utilise la loi de Dirichlet pour garantir que la somme des isotopes est de 100%.
    """
    np.random.seed(random_seed)
    isotope_columns = ['Pu-238/Pu', 'Pu-239/Pu', 'Pu-240/Pu', 'Pu-241/Pu', 'Pu-242/Pu']
    
    # Vectorisation pour la rapidité
    ref_spectra = np.stack(reference_df["content"].values)
    ref_comps = reference_df[isotope_columns].values
    
    synthetic_data = []
    print(f"Generating {n_synthetic} synthetic spectra...")

    for i in range(n_synthetic):
        n_combine = np.random.randint(min_spectra, max_spectra + 1)
        # Sélection d'indices aléatoires
        indices = np.random.choice(len(reference_df), n_combine, replace=False)
        
        # Dirichlet génère des poids dont la somme est exactement 1.0
        weights = np.random.dirichlet([alpha] * n_combine)
        
        # Calcul du nouveau spectre et de la nouvelle composition
        combined_spec = np.sum(weights[:, np.newaxis] * ref_spectra[indices], axis=0)
        combined_comp = np.sum(weights[:, np.newaxis] * ref_comps[indices], axis=0)
        
        if add_noise:
            # Bruit de Poisson physique
            combined_spec = np.random.poisson(np.maximum(combined_spec, 0))
        
        # On hérite des métadonnées du premier parent
        parent = reference_df.iloc[indices[0]]
        
        row = {
            'File name': f'synth_{i}.txt',
            'Detector': parent['Detector'],
            'Detector quanti': parent['Detector quanti'],
            'content': combined_spec.astype(int),
            'real counting times': parent['real counting times'],
            'live counting times': parent['live counting times'],
            'FWHM at 208 keV (keV)': parent['FWHM at 208 keV (keV)'],
            'Am-241/Pu': parent['Am-241/Pu'],
            'Pu-238/Pu': combined_comp[0],
            'Pu-239/Pu': combined_comp[1],
            'Pu-240/Pu': combined_comp[2],
            'Pu-241/Pu': combined_comp[3],
            'Pu-242/Pu': combined_comp[4],
            'n_combined': n_combine
        }
        synthetic_data.append(row)
        
    return pd.DataFrame(synthetic_data)

# ==========================================
# 4. SPLIT ET NORMALISATION
# ==========================================

def stratified_train_val_split(df, train_ratio=0.8, random_seed=4):
    """Sépare les données en gardant l'équilibre entre détecteurs."""
    train_list, val_list = [], []
    for det in df['Detector quanti'].unique():
        subset = df[df['Detector quanti'] == det]
        train_sub = subset.sample(frac=train_ratio, random_state=random_seed)
        val_sub = subset.drop(train_sub.index)
        train_list.append(train_sub)
        val_list.append(val_sub)
    return pd.concat(train_list).reset_index(drop=True), pd.concat(val_list).reset_index(drop=True)

def scale_max_amplitude(train_df, val_df, test_df):
    """
    Ramène la colonne max_amplitude_log entre 0 et 1.
    """
    # 1. Calcul des bornes sur le train uniquement (pour éviter le data leakage)
    min_val = train_df["max_amplitude_log"].min()
    max_val = train_df["max_amplitude_log"].max()
    
    def apply_min_max(df):
        df = df.copy()
        # Formule : (x - min) / (max - min)
        df["max_amplitude_log_norm"] = (df["max_amplitude_log"] - min_val) / (max_val - min_val)
        return df

    train_scaled = apply_min_max(train_df)
    val_scaled = apply_min_max(val_df)
    test_scaled = apply_min_max(test_df)
    
    return train_scaled, val_scaled, test_scaled

def normalize_spectra(train_df, val_df, test_df, save_path="./"):
    """
    Normalise les spectres et prépare les métadonnées pour le VAE et le régresseur.
    Divise les pourcentages par 100 et conserve toutes les colonnes métier.
    """
    isotope_columns = ["Pu-238/Pu", "Pu-239/Pu", "Pu-240/Pu", "Pu-241/Pu", "Pu-242/Pu", "Am-241/Pu"]
    
    def process_dataframe(df):
        # 1. Extraction et transformation Log des spectres
        spectra = np.stack(df["content"].values)
        log_spectra = np.log1p(spectra)
        
        # 2. Calcul de l'amplitude max par signal
        max_amp = np.max(log_spectra, axis=1, keepdims=True)
        
        # 3. Normalisation du spectre (0 à 1 localement)
        norm_spectra = log_spectra / (max_amp + 1e-9)
        
        # 4. Création d'un nouveau DataFrame pour ne pas écraser l'original
        new_df = df.copy()
        
        # 5. Mise à jour des spectres normalisés
        new_df["content"] = list(norm_spectra)
        
        # 6. Ajout de l'amplitude maximale (utile pour reconstruire le signal réel)
        new_df["max_amplitude_log"] = max_amp.flatten()
        
        # 7. Division des isotopes par 100 (Plage 0-1)
        for col in isotope_columns:
            if col in new_df.columns:
                new_df[col] = new_df[col].astype(float) / 100.0
                
        return new_df

    # Application du traitement sur les 3 sets
    train_final = process_dataframe(train_df)
    val_final = process_dataframe(val_df)
    test_final = process_dataframe(test_df)

    # 8. Calcul des bornes globales pour archive
    global_max = float(max(train_final["max_amplitude_log"].max(), val_final["max_amplitude_log"].max()))
    global_min = float(min(train_final["max_amplitude_log"].min(), val_final["max_amplitude_log"].min()))

    # Sauvegarde des paramètres de normalisation
    norm_metadata = {
        "global_max_log_amp": global_max,
        "global_min_log_amp": global_min,
        "isotope_scaling": "divided_by_100",
        "spectrum_scaling": "log1p_then_per_sample_max_scaling",
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    with open(os.path.join(save_path, "normalization_metadata.json"), 'w') as f:
        json.dump(norm_metadata, f, indent=2)

    print(f"Normalization completed. Isotope values are now in range [0, 1].")
    print(f"Max log amplitude saved in 'max_amplitude_log' column.")
    
    return (train_final, val_final, test_final), norm_metadata

# ==========================================
# 5. MAIN EXECUTION
# ==========================================

def save_datasets(datasets, filenames):
    """
    Sauvegarde les datasets. Gère les DataFrames et les arrays NumPy.
    """
    for dataset, filename in zip(datasets, filenames):
        if isinstance(dataset, pd.DataFrame):
            dataset.to_json(filename)
        elif isinstance(dataset, np.ndarray):
            # Pour les matrices normalisées (X_train, etc.)
            # On peut les sauver en JSON via une conversion en liste ou en .npy
            pd.DataFrame(dataset).to_json(filename) 
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
    create_histograms(df_clean,save_path="./before")
    
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

    # 1. Génération (Vérifie que alpha est passé si tu veux contrôler le mélange)
    augmented_dataset = generate_linear_combination_spectra_constrained(
        train_dataset, 
        n_synthetic=5000, 
        min_spectra=2, 
        max_spectra=10,
        random_seed=42, 
        add_noise=True,
        alpha=0.5 # Ajout du paramètre pour le simplexe
    )

    # 2. Sauvegarde des sets complets (DataFrames avec isotopes)
    save_datasets(
        (augmented_dataset, val_dataset, test_dataset),
        ["train_aug_meta.json", "val_meta.json", "test_meta.json"]
    )
    
    # 3. Normalisation
    # On récupère les matrices de spectres (X) et les métadonnées (config)
    (train_norm, val_norm, test_norm), norm_metadata = normalize_spectra(
        augmented_dataset, val_dataset, test_dataset, save_path="./"
    )
    
    train_norm, val_norm, test_norm = scale_max_amplitude(train_norm, val_norm, test_norm)

    # 4. ATTENTION : create_histograms prend un DataFrame. 
    # Si tu veux voir la distri des isotopes après augmentation :
    create_histograms(augmented_dataset, save_path="./augmented2_")
 
    # 5. Sauvegarde finale des matrices d'entrée pour le VAE
    # Ce sont ces fichiers que ton script d'entraînement lira
    save_datasets(
        (train_norm, val_norm, test_norm),
        ["X_train.json", "X_val.json", "X_test.json"]
    )

    print("\n=== Processing completed successfully ===")

if __name__ == "__main__":
    main()