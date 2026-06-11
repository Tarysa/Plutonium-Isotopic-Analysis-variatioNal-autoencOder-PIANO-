import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import json
from sklearn.cluster import KMeans
from collections import defaultdict
from scipy.stats import gaussian_kde
from scipy.optimize import minimize
from scipy.interpolate import PchipInterpolator
import matplotlib.ticker as ticker
import seaborn as sns

E_common = np.linspace(0, 3.0, 4096)


# =============================================================================
# UTILITAIRES
# =============================================================================

def resize_spectrum_spline(spectrum, target_size):
    x_original = np.arange(len(spectrum))
    x_new = np.linspace(0, len(spectrum) - 1, target_size)
    interp = PchipInterpolator(x_original, spectrum)
    return interp(x_new)


def indices_par_nom(L):
    d = defaultdict(list)
    for i, nom in enumerate(L):
        d[nom].append(i)
    return dict(d)


# =============================================================================
# PARSING ET CHARGEMENT
# =============================================================================

def parse_spectrum_file(file_path):
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
    while data_not_found:
        ligne = file_lines[i]
        if "$DATA" in ligne:
            data_not_found = False
        else:
            ligne = ligne.strip()
        if not ligne:
            continue
        if ":" in ligne:
            key, value = ligne.split(":", 1)
        else:
            key, value = ligne.split(None, 1)
        metadata_dict[key.strip()] = value.strip().split(" ")[0]
        i += 1

    nb_line = int(file_lines[i].split(" ")[1]) + 1

    if "ENER_FIT" in file_lines[i + nb_line]:
        spectrum = [0]
        for j in range(nb_line - 1):
            spectrum.append(int(file_lines[i + j + 1]))
        spectrum = np.array(spectrum)
        id_to_continue = i + nb_line + 1
    else:
        spectrum = []
        for j in range(nb_line):
            spectrum.append(int(file_lines[i + j + 1]))
        spectrum = np.array(spectrum)
        id_to_continue = i + nb_line + 2

    spectrum = resize_spectrum_spline(spectrum, 4096)
    metadata_dict["content"] = [spectrum]
    metadata_dict["ENER_FIT"] = file_lines[id_to_continue].strip().split(" ")[1]

    for j in range(id_to_continue + 1, len(file_lines)):
        ligne = file_lines[j].strip()
        if not ligne:
            continue
        try:
            if ":" in ligne:
                key, value = ligne.split(":", 1)
            else:
                key, value = ligne.split(None, 1)
            metadata_dict[key.strip()] = value.strip().split(" ")[0]
        except Exception:
            continue

    key_to_remove = [k for k, v in metadata_dict.items() if v is None or v == ""]
    for k in key_to_remove:
        del metadata_dict[k]

    return metadata_dict


def process_metadata(metadata_dict):
    expected_columns = [
        "File name", "Detector",
        "234U", "235U", "236U", "238U",
        "content", "ENER_FIT"
    ]
    cols_U  = ["234U",  "235U",  "236U",  "238U"]

    sum_U  = 0.0

    processed_dict = {}
    for col in expected_columns:
        if col in metadata_dict:
            if metadata_dict[col] != "Unknown":
                processed_dict[col] = metadata_dict[col]
                if col in cols_U:
                    sum_U  += float(metadata_dict[col])
            else:
                processed_dict[col] = 0
        else:
            processed_dict[col] = -100

    col_U_sums  = 99 < sum_U  < 101

    if col_U_sums:
        return processed_dict
    else:
        print("Sum not equal to one")
        return None


def plot_row_sum_histogram(df, columns_to_sum, num_bins=20, save_path="only_U/"):
    temp_df = df[columns_to_sum].apply(pd.to_numeric, errors='coerce').dropna()
    row_sums = temp_df.sum(axis=1)
    plt.figure(figsize=(12, 7))
    ax = sns.histplot(row_sums, bins=num_bins, kde=True, color='royalblue', edgecolor='white')
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%g'))
    plt.xticks(rotation=45)
    plt.title('Distribution des sommes')
    plt.xlabel('Somme calculée')
    plt.ylabel('Fréquence')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{save_path}hist_sum.png")
    plt.close()


def load_and_process_spectra(base_path):
    dataframes = []
    for filename in os.listdir(f"{base_path}/IDB-all-spectra/"):
        print(f"  Processing file: {filename}")
        file_path = f"{base_path}/IDB-all-spectra/{filename}"
        metadata_dict = parse_spectrum_file(file_path)
        if metadata_dict is None:
            continue
        try:
            processed_metadata = process_metadata(metadata_dict)
            if processed_metadata is not None:
                df_file = pd.DataFrame.from_dict(processed_metadata)
                dataframes.append(df_file)
        except ValueError as e:
            print(f"Error in file {filename}: {e}")

    print(len(dataframes))
    final_df = pd.concat(dataframes, ignore_index=True, join="inner")
    print(len(final_df))
    final_df = final_df.dropna()
    return final_df


# =============================================================================
# NETTOYAGE ET FILTRAGE
# =============================================================================

def clean_and_convert_data(df):
    df = df[~df["Detector"].isin(["NaI", "CdTe"])]

    all_iso = ["234U", "235U", "236U", "238U"]
    for col in all_iso:
        if col in df.columns:
            df[col] = df[col].astype(float)
    if 'ENER_FIT' in df.columns:
        df['ENER_FIT'] = df['ENER_FIT'].astype(float)

    # Filtres physiques
    df = df[df["234U"]  < 20]
    df = df[df["235U"]  > 60]
    df = df[df["236U"]  < 40]

    return df


def filter_and_normalize_data(df):
    filtered_df = df.copy()
    columns_to_keep = [
        "File name", "Detector",
        "234U", "235U", "236U", "238U",
        "241Am", "content", "ENER_FIT"
    ]
    available_columns = [col for col in columns_to_keep if col in filtered_df.columns]
    return filtered_df[available_columns].copy()


# =============================================================================
# CLUSTERING
# =============================================================================

def detect_isotope_clusters(df, isotope_columns, n_clusters=3, random_seed=42):
    """KMeans sur l'espace isotopique pour identifier les modes discrets."""
    data = df[isotope_columns].values.astype(float)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
    labels = kmeans.fit_predict(data)

    print(f"\n=== Détection de {n_clusters} clusters ===")
    for c in range(n_clusters):
        mask = labels == c
        print(f"  Cluster {c}: {mask.sum()} échantillons")
        for iso in isotope_columns:
            vals = df[iso].values[mask]
            print(f"    {iso}: mean={vals.mean():.4f}, std={vals.std():.4f}")

    return labels, kmeans


# =============================================================================
# BORNES
# =============================================================================

def compute_isotope_bounds(reference_df, isotope_columns, margin=0.0):
    bounds = {}
    for isotope in isotope_columns:
        min_val = float(reference_df[isotope].min())
        max_val = float(reference_df[isotope].max())
        if margin > 0:
            range_val = max_val - min_val
            min_val = max(0.0,   min_val - margin * range_val)
            max_val = min(100.0, max_val + margin * range_val)
        bounds[isotope] = (min_val, max_val)
    print("\n=== Isotope Bounds ===")
    for iso, (lo, hi) in bounds.items():
        print(f"  {iso}: [{lo:.4f}, {hi:.4f}]")
    return bounds


def compute_isotope_bounds_per_cluster(clean_df, cluster_labels, isotope_columns,
                                       n_clusters, margin=0.0):
    isotope_bounds_per_cluster = {}
    for c in range(n_clusters):
        mask = cluster_labels == c
        print(f"\n--- Bornes cluster {c} ({mask.sum()} échantillons) ---")
        isotope_bounds_per_cluster[c] = compute_isotope_bounds(
            clean_df[mask], isotope_columns, margin
        )
    return isotope_bounds_per_cluster


# =============================================================================
# GÉNÉRATION DE POIDS CONTRAINTS
# =============================================================================

def is_composition_valid(composition, bounds, isotope_columns, tolerance=1e-6):
    total = composition.sum()
    if abs(total - 100.0) > tolerance:
        return False
    for i, iso in enumerate(isotope_columns):
        lo, hi = bounds[iso]
        if composition[i] < lo or composition[i] > hi:
            return False
    return True


def generate_constrained_weights(ref_compositions, selected_indices, bounds, isotope_columns):
    """
    Retourne des poids valides pour la combinaison linéaire.
    Stratégie : 30 tirages Dirichlet → SLSQP(maxiter=1000) → fallback uniforme.
    """
    n_combine = len(selected_indices)
    selected_comps = ref_compositions[selected_indices]

    # --- Tentatives rapides Dirichlet ---
    for _ in range(30):
        w = np.random.dirichlet(0.2 * np.ones(n_combine))
        combined = np.sum(w[:, np.newaxis] * selected_comps, axis=0)
        combined_norm = (combined / combined.sum()) * 100.0
        if is_composition_valid(combined_norm, bounds, isotope_columns):
            return w

    # --- Optimisation SLSQP ---
    def objective(w):
        return np.sum((w - 1.0 / n_combine) ** 2)

    def constraint_sum(w):
        return np.sum(w) - 1.0

    def make_bound_constraints(iso_idx):
        lo, hi = bounds[isotope_columns[iso_idx]]
        def c_min(w):
            val = np.sum(w * selected_comps[:, iso_idx])
            tot = np.sum(w[:, np.newaxis] * selected_comps, axis=0).sum()
            return (val / tot) * 100.0 - lo
        def c_max(w):
            val = np.sum(w * selected_comps[:, iso_idx])
            tot = np.sum(w[:, np.newaxis] * selected_comps, axis=0).sum()
            return hi - (val / tot) * 100.0
        return c_min, c_max

    constraints = [{'type': 'eq', 'fun': constraint_sum}]
    for i in range(len(isotope_columns)):
        c_min, c_max = make_bound_constraints(i)
        constraints += [
            {'type': 'ineq', 'fun': c_min},
            {'type': 'ineq', 'fun': c_max},
        ]

    w0 = np.ones(n_combine) / n_combine
    result = minimize(
        objective, w0,
        method='SLSQP',
        bounds=[(0.01, 0.99)] * n_combine,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-6}
    )
    if result.success:
        return result.x / result.x.sum()

    # --- Fallback uniforme ---
    uniform = np.ones(n_combine) / n_combine
    combined = np.sum(uniform[:, np.newaxis] * selected_comps, axis=0)
    print(f"  [fallback] {dict(zip(isotope_columns, (combined/combined.sum()*100).round(3)))}")
    return uniform


# =============================================================================
# GÉNÉRATION UNIFORME SUR L'ESPACE ISOTOPIQUE
# (couvre les zones peu représentées — clé pour l'uranium)
# =============================================================================

def generate_compositions_uniform(reference_df, isotope_columns, n_synthetic=1000,
                                  random_seed=42):
    """
    Tire des compositions uniformément sur le simplex via Dirichlet(1,1,...,1)
    puis rejette celles qui sortent des bornes observées.
    Garantit une couverture continue de l'espace — complémentaire aux
    combinaisons intra-cluster qui restent collées aux modes discrets.
    """
    np.random.seed(random_seed)
    bounds = compute_isotope_bounds(reference_df, isotope_columns)
    n_iso  = len(isotope_columns)

    compositions = []
    attempts     = 0
    max_attempts = n_synthetic * 200   # sécurité anti-boucle infinie

    while len(compositions) < n_synthetic and attempts < max_attempts:
        attempts += 1
        sample = np.random.dirichlet(np.ones(n_iso)) * 100.0
        valid  = all(
            bounds[iso][0] <= sample[i] <= bounds[iso][1]
            for i, iso in enumerate(isotope_columns)
        )
        if valid:
            compositions.append(sample)

    print(f"Uniform sampling: {len(compositions)}/{n_synthetic} en {attempts} tentatives "
          f"(taux={100*len(compositions)/max(attempts,1):.1f}%)")
    return np.array(compositions)


def generate_physically_consistent_spectra(target_compositions, reference_df,
                                           isotope_columns, ref_spectra, ref_compositions,
                                           detectors, add_noise=True, random_seed=42,
                                           n_pairs_tried=300):
    """
    Pour chaque composition cible, trouve la paire de spectres réels dont
    la combinaison linéaire reproduit cette composition (résolution sur 235U
    ou l'isotope principal), puis combine les spectres avec les mêmes poids.

    => Garantit la cohérence spectre <-> composition, contrairement à
       assign_spectrum_to_composition qui colle un faux label sur un spectre réel.
    """
    np.random.seed(random_seed)

    # Indice de l'isotope principal (celui avec la plus grande variance → 235U ou 239Pu)
    variances  = ref_compositions.var(axis=0)
    pivot_idx  = int(np.argmax(variances))
    pivot_name = isotope_columns[pivot_idx]
    print(f"  Pivot isotope pour interpolation : {pivot_name}")

    n_ref = len(ref_compositions)
    synthetic_data = []
    failed = 0

    for i, target in enumerate(target_compositions):
        target_pivot = target[pivot_idx]
        best = None
        best_residual = np.inf

        for _ in range(n_pairs_tried):
            idx_A, idx_B = np.random.choice(n_ref, size=2, replace=False)

            # Même détecteur obligatoire pour cohérence physique
            if detectors[idx_A] != detectors[idx_B]:
                continue

            pA = ref_compositions[idx_A, pivot_idx]
            pB = ref_compositions[idx_B, pivot_idx]
            denom = pA - pB
            if abs(denom) < 0.5:   # trop proches → combinaison instable
                continue

            alpha = (target_pivot - pB) / denom
            if not (0.05 <= alpha <= 0.95):
                continue

            combined      = alpha * ref_compositions[idx_A] + (1 - alpha) * ref_compositions[idx_B]
            combined_norm = (combined / combined.sum()) * 100.0
            residual      = np.linalg.norm(combined_norm - target)

            if residual < best_residual:
                best_residual = residual
                best = (idx_A, idx_B, alpha)

        # Seuil de résidu acceptable (2% absolu)
        if best is None or best_residual > 2.0:
            failed += 1
            continue

        idx_A, idx_B, alpha = best

        # Spectre combiné avec les mêmes poids → cohérent avec la composition
        spectrum = alpha * ref_spectra[idx_A] + (1 - alpha) * ref_spectra[idx_B]
        if add_noise:
            spectrum = np.maximum(spectrum, 0)
            spectrum = np.random.poisson(spectrum)

        # Composition réelle (pas target, pour rester honnête)
        final_comp = alpha * ref_compositions[idx_A] + (1 - alpha) * ref_compositions[idx_B]
        final_comp = (final_comp / final_comp.sum()) * 100.0

        row = {
            'File name':          f'synthetic_uniform_{i:04d}.txt',
            'Detector':           detectors[idx_A],
            'content':            spectrum.astype(int),
            'ENER_FIT':           reference_df.iloc[idx_A]['ENER_FIT'],
            'cluster':            'uniform_physical',
            'n_combined_spectra': 2,
            'combination_weights': [float(alpha), float(1 - alpha)],
            'combined_indices':    [int(idx_A), int(idx_B)],
        }
        for k, iso in enumerate(isotope_columns):
            row[iso] = final_comp[k]

        synthetic_data.append(row)

    print(f"Physically consistent: {len(synthetic_data)} générés, {failed} échecs")
    return pd.DataFrame(synthetic_data)


# =============================================================================
# GÉNÉRATION INTRA-CLUSTER (combinaison linéaire contrainte par cluster)
# =============================================================================

def generate_linear_combination_spectra_constrained(
    reference_df,
    isotope_columns,           # ['234U','235U','236U','238U'] ou ['238Pu',...]
    n_synthetic=1000,
    min_spectra=2,
    max_spectra=4,
    random_seed=42,
    add_noise=True,
    isotope_margin=0.0,
    max_retries=50,
    n_clusters=3,
    cluster_quota=None,
    use_density_sampling=True, # hérité du code Pu qui a bien fonctionné
):
    """
    Combinaison linéaire contrainte, stratifiée par cluster KMeans.
    Copie la logique du code Pu (density sampling) + fixes U :
      - bornes par cluster
      - replace=True si cluster trop petit
      - n_combine plafonné
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    print(f"\n=== Generating {n_synthetic} intra-cluster samples "
          f"[{', '.join(isotope_columns)}] ===")

    # --- Sélection des spectres avec composition valide ---
    clean_df = reference_df[
        (reference_df[isotope_columns] != -100).all(axis=1)
    ].copy()

    sum_ = clean_df[isotope_columns].sum(axis=1)
    frac = sum_.between(99, 101).sum() / len(clean_df)
    print(f"Fraction valide : {frac:.3f}  ({len(clean_df)} spectres)")

    ref_spectra   = np.array([s for s in clean_df["content"].values])
    ref_comps     = clean_df[isotope_columns].values.astype(float)
    detectors     = clean_df["Detector"].values
    detector_quanti_dict = dict(zip(clean_df["Detector"],
                                    clean_df["Detector quanti"]))

    # --- Clustering ---
    cluster_labels, _ = detect_isotope_clusters(
        clean_df, isotope_columns, n_clusters=n_clusters, random_seed=random_seed
    )
    clean_df["_cluster"] = cluster_labels

    # --- Bornes PAR cluster ---
    bounds_per_cluster = compute_isotope_bounds_per_cluster(
        clean_df, cluster_labels, isotope_columns, n_clusters, isotope_margin
    )

    # --- Quota équiréparti avec correction d'arrondi ---
    if cluster_quota is None:
        cluster_quota = {c: n_synthetic // n_clusters for c in range(n_clusters)}
        diff = n_synthetic - sum(cluster_quota.values())
        cluster_quota[0] += diff

    print("\n=== Quota par cluster ===")
    for c, q in cluster_quota.items():
        print(f"  Cluster {c}: {q}")

    # --- Index par cluster ET par détecteur ---
    cluster_det_idx = {}
    for c in range(n_clusters):
        g_idx    = np.where(cluster_labels == c)[0]
        det_names = [detectors[i] for i in g_idx]
        local_map = indices_par_nom(det_names)
        cluster_det_idx[c] = {
            det: g_idx[np.array(idxs)]
            for det, idxs in local_map.items()
        }

    # --- Density sampling (comme le code Pu) ---
    detector_dict      = indices_par_nom(detectors)
    name_detector      = np.unique(detectors)
    det_common_idx     = {}
    det_rare_idx       = {}
    det_inv_dens       = {}

    if use_density_sampling:
        print("\n=== Computing density ===")
        iso_vals = ref_comps.T
        kde      = gaussian_kde(iso_vals)
        dens     = kde(iso_vals)
        for name in name_detector:
            mask_det = np.array(detector_dict[name])
            low_mask = dens[mask_det] < np.percentile(dens[mask_det], 40)
            det_rare_idx[name]   = mask_det[np.where(low_mask)[0]]
            det_common_idx[name] = mask_det[np.where(~low_mask)[0]]
            inv_d = 1.0 / (dens[mask_det] + 1e-6)
            det_inv_dens[name] = inv_d / inv_d.sum()

    # --- Boucle de génération stratifiée par cluster ---
    synthetic_data  = []
    failed_attempts = 0
    total_attempts  = 0

    for cluster_id, quota in cluster_quota.items():
        print(f"\n--- Cluster {cluster_id} ({quota} samples) ---")
        avail_by_det = cluster_det_idx[cluster_id]
        all_avail    = np.concatenate(list(avail_by_det.values())) if avail_by_det else np.array([])

        if len(all_avail) == 0:
            print(f"  Cluster {cluster_id} vide, skip.")
            continue

        bounds     = bounds_per_cluster[cluster_id]
        sample_idx = 0

        while sample_idx < quota:
            # Détecteurs avec ≥ 1 spectre dans ce cluster
            avail_dets = [d for d, idxs in avail_by_det.items() if len(idxs) >= 1]
            if not avail_dets:
                print(f"  Cluster {cluster_id} sans détecteur, skip.")
                break

            detector_name    = avail_dets[np.random.randint(len(avail_dets))]
            avail_indices    = avail_by_det[detector_name]

            # Plafonne n_combine pour les petits clusters
            max_combine = min(max_spectra, max(2, len(avail_indices) // 2 + 1))
            n_combine   = np.random.randint(min_spectra, max_combine + 1)

            success = False
            for _ in range(max_retries):
                total_attempts += 1

                # Density sampling à l'intérieur du cluster
                if use_density_sampling and detector_name in det_inv_dens:
                    # Filtrer les probabilités sur les indices du cluster
                    cluster_mask_for_det = np.isin(
                        np.array(detector_dict[detector_name]), avail_indices
                    )
                    local_avail = np.array(detector_dict[detector_name])[cluster_mask_for_det]
                    local_probs = det_inv_dens[detector_name][cluster_mask_for_det]

                    if len(local_avail) >= n_combine and local_probs.sum() > 0:
                        local_probs = local_probs / local_probs.sum()
                        replace_    = len(local_avail) < n_combine
                        selected_indices = np.random.choice(
                            local_avail, size=n_combine,
                            replace=replace_, p=local_probs
                        )
                    else:
                        replace_ = len(avail_indices) < n_combine
                        selected_indices = np.random.choice(
                            avail_indices, size=n_combine, replace=replace_
                        )
                else:
                    replace_ = len(avail_indices) < n_combine
                    selected_indices = np.random.choice(
                        avail_indices, size=n_combine, replace=replace_
                    )

                weights = generate_constrained_weights(
                    ref_comps, selected_indices, bounds, isotope_columns
                )
                if weights is None:
                    failed_attempts += 1
                    continue

                # Combinaison spectrale
                combined_spectrum = np.zeros_like(ref_spectra[0], dtype=float)
                for idx, w in zip(selected_indices, weights):
                    combined_spectrum += w * ref_spectra[idx]

                if add_noise:
                    combined_spectrum = np.maximum(combined_spectrum, 0)
                    combined_spectrum = np.random.poisson(combined_spectrum)
                combined_spectrum = combined_spectrum.astype(int)

                # Composition combinée — normalisée sur les isotopes cibles UNIQUEMENT
                combined_comp = np.zeros(len(isotope_columns), dtype=float)
                for idx, w in zip(selected_indices, weights):
                    combined_comp += w * ref_comps[idx]

                comp_sum = combined_comp.sum()
                if comp_sum <= 0:
                    continue
                combined_comp = (combined_comp / comp_sum) * 100.0

                if not (99 <= combined_comp.sum() <= 101):
                    continue

                ref_row = clean_df.iloc[selected_indices[0]]
                row = {
                    'File name':          f'synthetic_c{cluster_id}_{sample_idx:04d}.txt',
                    'Detector':           detector_name,
                    'Detector quanti':    detector_quanti_dict[detector_name],
                    'content':            combined_spectrum,
                    'ENER_FIT':           ref_row['ENER_FIT'],
                    'cluster':            cluster_id,
                    'n_combined_spectra': n_combine,
                    'combination_weights': weights.tolist(),
                    'combined_indices':   selected_indices.tolist(),
                }
                for k, iso in enumerate(isotope_columns):
                    row[iso] = combined_comp[k]

                synthetic_data.append(row)
                sample_idx += 1
                success = True
                break

            if not success:
                print(f"  Warning: échec sample {sample_idx} cluster {cluster_id}")
                sample_idx += 1

            if sample_idx % 100 == 0 and sample_idx > 0:
                print(f"  [{cluster_id}] {sample_idx}/{quota}")

    synthetic_df = pd.DataFrame(synthetic_data)

    # --- Validation ---
    print(f"\n=== Validation intra-cluster ===")
    comp_sums = synthetic_df[isotope_columns].sum(axis=1)
    print(f"Composition sum — Mean: {comp_sums.mean():.4f}%, Std: {comp_sums.std():.2e}%")
    for c in range(n_clusters):
        cs = synthetic_df[synthetic_df["cluster"] == c]
        if len(cs) == 0:
            continue
        print(f"\n  Cluster {c}:")
        for iso in isotope_columns:
            lo, hi = bounds_per_cluster[c][iso]
            a_min, a_max = cs[iso].min(), cs[iso].max()
            ok = (a_min >= lo - 1e-6) and (a_max <= hi + 1e-6)
            print(f"    {iso}: [{a_min:.4f}, {a_max:.4f}] vs [{lo:.4f}, {hi:.4f}] "
                  f"{'✓' if ok else '✗'}")
    print(f"Success: {len(synthetic_data)}/{total_attempts} "
          f"({100*len(synthetic_data)/max(total_attempts,1):.1f}%)")

    return synthetic_df


# =============================================================================
# GÉNÉRATION INTER-CLUSTER
# =============================================================================

def generate_intercluster_spectra(
    clean_reference_df, cluster_labels, isotope_bounds_per_cluster,
    ref_spectra, ref_compositions, detectors, isotope_columns,
    n_synthetic=300, cluster_pairs=None, add_noise=True, random_seed=44,
):
    np.random.seed(random_seed)
    n_clusters = len(np.unique(cluster_labels))

    if cluster_pairs is None:
        cluster_pairs = [(a, b) for a in range(n_clusters)
                         for b in range(a + 1, n_clusters)]

    n_per_pair = n_synthetic // len(cluster_pairs)
    remainder  = n_synthetic - n_per_pair * len(cluster_pairs)

    print(f"\n=== Génération inter-clusters ===  paires={cluster_pairs}")
    print(f"  {n_per_pair} samples/paire ({remainder} extra sur paire 0)")

    idx_by_cluster = {c: np.where(cluster_labels == c)[0] for c in range(n_clusters)}
    detector_quanti_dict = dict(zip(
        clean_reference_df["Detector"], clean_reference_df["Detector quanti"]
    ))

    synthetic_data = []
    total_attempts = 0

    for pair_idx, (cA, cB) in enumerate(cluster_pairs):
        quota   = n_per_pair + (remainder if pair_idx == 0 else 0)
        idxs_A  = idx_by_cluster[cA]
        idxs_B  = idx_by_cluster[cB]
        dets_A  = detectors[idxs_A]
        dets_B  = detectors[idxs_B]
        common  = list(set(dets_A) & set(dets_B))

        print(f"\n  Paire ({cA},{cB}) — {quota} samples  détecteurs communs: {common}")
        if not common:
            print("  Skip : aucun détecteur commun.")
            continue

        sample_idx = 0
        while sample_idx < quota:
            total_attempts += 1
            if total_attempts > quota * 20:
                print(f"  Abandon paire ({cA},{cB}) après trop d'échecs.")
                break

            det    = common[np.random.randint(len(common))]
            av_A   = idxs_A[dets_A == det]
            av_B   = idxs_B[dets_B == det]
            if len(av_A) == 0 or len(av_B) == 0:
                continue

            idx_A = np.random.choice(av_A)
            idx_B = np.random.choice(av_B)
            alpha = np.random.uniform(0.1, 0.9)

            comp_A = ref_compositions[idx_A].astype(float)
            comp_B = ref_compositions[idx_B].astype(float)
            combined_comp = alpha * comp_A + (1 - alpha) * comp_B
            s = combined_comp.sum()
            if s > 0:
                combined_comp = (combined_comp / s) * 100.0

            # Rejeter si tombe entièrement dans un cluster source (80% du temps)
            def in_cluster(c, comp):
                return all(
                    isotope_bounds_per_cluster[c][iso][0] <= comp[i]
                    <= isotope_bounds_per_cluster[c][iso][1]
                    for i, iso in enumerate(isotope_columns)
                )
            if (in_cluster(cA, combined_comp) or in_cluster(cB, combined_comp)):
                if np.random.rand() > 0.2:
                    continue

            spectrum = alpha * ref_spectra[idx_A] + (1 - alpha) * ref_spectra[idx_B]
            if add_noise:
                spectrum = np.maximum(spectrum, 0)
                spectrum = np.random.poisson(spectrum)

            ref_row = clean_reference_df.iloc[idx_A]
            row = {
                'File name':          f'synthetic_inter_{cA}{cB}_{sample_idx:04d}.txt',
                'Detector':           det,
                'Detector quanti':    detector_quanti_dict.get(det, -1),
                'content':            spectrum.astype(int),
                'ENER_FIT':           ref_row['ENER_FIT'],
                'cluster':            f'inter_{cA}{cB}',
                'n_combined_spectra': 2,
                'combination_weights': [float(alpha), float(1 - alpha)],
                'combined_indices':    [int(idx_A), int(idx_B)],
            }
            for k, iso in enumerate(isotope_columns):
                row[iso] = combined_comp[k]

            synthetic_data.append(row)
            sample_idx += 1

        print(f"  Paire ({cA},{cB}) : {sample_idx} générés")

    print(f"\n  Total inter: {len(synthetic_data)}/{total_attempts} "
          f"({100*len(synthetic_data)/max(total_attempts,1):.1f}%)")
    return pd.DataFrame(synthetic_data)


# =============================================================================
# VISUALISATION
# =============================================================================

def plot_synthetic_vs_real_coverage(real_df, synthetic_df, isotope_columns, save_path):
    n = len(isotope_columns)
    fig, axes = plt.subplots(n, n, figsize=(14, 14))
    for i, ix in enumerate(isotope_columns):
        for j, iy in enumerate(isotope_columns):
            ax = axes[i][j]
            if i == j:
                ax.hist(real_df[ix],      bins=30, alpha=0.5, color='blue',
                        label='Real',      density=True)
                ax.hist(synthetic_df[ix], bins=30, alpha=0.5, color='orange',
                        label='Synthetic', density=True)
            else:
                ax.scatter(real_df[ix],      real_df[iy],      s=5,  alpha=0.4,
                           color='blue')
                ax.scatter(synthetic_df[ix], synthetic_df[iy], s=5,  alpha=0.2,
                           color='orange')
            if i == 0: ax.set_title(iy, fontsize=8)
            if j == 0: ax.set_ylabel(ix, fontsize=8)
    axes[0][-1].legend(fontsize=7)
    plt.suptitle("Couverture réel vs synthétique", fontsize=12)
    plt.tight_layout()
    plt.savefig(f"{save_path}coverage_real_vs_synthetic.png", dpi=150)
    plt.close()
    print("Coverage plot saved.")


def create_histograms(df, save_path="only_U/", normalized=False):
    exclude = ['File name', 'Detector', 'content', 'combination_weights', 'combined_indices']
    for column in df.columns:
        if column in exclude:
            continue
        suffix = "_normalized" if normalized else ""
        fname  = f"{save_path}hist_{column.replace(' ','_').replace('/','_')}{suffix}.png"
        try:
            plt.figure(figsize=(10, 6))
            plt.hist(df[df[column] >= 0][[column]], bins=30,
                     edgecolor="black", alpha=0.7, color='skyblue')
            plt.xlabel(column); plt.ylabel("Frequency")
            plt.title(f"Distribution of {column}")
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.savefig(fname, dpi=300)
            plt.close()
        except Exception:
            plt.close()

def plot_isotope_dist_by_detector(df, isotope_columns, save_path="only_U/"):
    """
    Génère des boxplots montrant la distribution de chaque isotope 
    en fonction du détecteur.
    """
    # On prépare la figure : 1 ligne par isotope
    n_iso = len(isotope_columns)
    fig, axes = plt.subplots(n_iso, 1, figsize=(12, 5 * n_iso))
    
    # Si un seul isotope, axes n'est pas une liste
    if n_iso == 1:
        axes = [axes]

    for i, iso in enumerate(isotope_columns):
        sns.boxplot(
            data=df, 
            x='Detector', 
            y=iso, 
            ax=axes[i], 
            palette="Set3", 
            hue='Detector', 
            legend=False
        )
        # Optionnel : ajouter les points individuels pour voir la densité (stripplot)
        sns.stripplot(
            data=df, 
            x='Detector', 
            y=iso, 
            ax=axes[i], 
            color="black", 
            size=3, 
            alpha=0.3
        )
        
        axes[i].set_title(f"Distribution de l'isotope {iso} par Détecteur", fontsize=14)
        axes[i].set_xlabel("Type de Détecteur")
        axes[i].set_ylabel("Concentration (%)")
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    
    # Créer le dossier si besoin
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    plt.savefig(f"{save_path}dist_isotopes_par_detecteur.png")
    

# =============================================================================
# SPLIT TRAIN / VAL
# =============================================================================

def stratified_train_val_split(df, train_ratio=0.8, random_seed=4):
    random.seed(random_seed)
    train_dataset = pd.DataFrame(columns=df.columns.tolist())
    val_dataset   = pd.DataFrame()

    df["Detector quanti"] = df["Detector"].astype("category").cat.codes
    detector_types        = df['Detector quanti'].value_counts().index
    print("Detector distribution:")
    print(df['Detector quanti'].value_counts())

    for dt in detector_types:
        subset  = df[df['Detector quanti'] == dt]
        indices = list(range(len(subset)))
        random.shuffle(indices)
        sp = int(np.floor(train_ratio * len(subset)))
        train_dataset = pd.concat([train_dataset, subset.iloc[indices[:sp]]],
                                  ignore_index=True)
        val_dataset   = pd.concat([val_dataset,   subset.iloc[indices[sp:]]],
                                  ignore_index=True)

    return train_dataset, val_dataset


# =============================================================================
# NORMALISATION
# =============================================================================

def normalize_spectra(train_df, val_df, test_df, isotope_columns_to_normalize,
                      save_path="only_U/"):
    train_sp = np.array([s for s in train_df["content"].values])
    val_sp   = np.array([s for s in val_df["content"].values])
    test_sp  = np.array([s for s in test_df["content"].values])

    train_log = np.log1p(train_sp)
    val_log   = np.log1p(val_sp)
    test_log  = np.log1p(test_sp)

    train_max = np.max(train_log, axis=1)[:, np.newaxis]
    val_max   = np.max(val_log,   axis=1)[:, np.newaxis]
    test_max  = np.max(test_log,  axis=1)[:, np.newaxis]

    global_max = max(np.max(train_max), np.max(val_max), np.max(test_max))
    global_min = min(np.min(train_max), np.min(val_max), np.min(test_max))

    train_norm       = train_log / train_max
    val_norm         = val_log   / val_max
    test_norm        = test_log  / test_max
    train_max_scaled = (train_max - global_min) / (global_max - global_min)
    val_max_scaled   = (val_max   - global_min) / (global_max - global_min)
    test_max_scaled  = (test_max  - global_min) / (global_max - global_min)

    meta = {
        "global_min": float(global_min), "global_max": float(global_max),
        "normalization_method": "log1p_then_max_amplitude_scaling",
        "timestamp": pd.Timestamp.now().isoformat(),
        "dataset_info": {
            "train_samples": len(train_df), "val_samples": len(val_df),
            "test_samples": len(test_df),
            "total_samples": len(train_df)+len(val_df)+len(test_df)
        },
    }
    meta_path = f"{save_path}/normalization_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved: {meta_path}")

    normalized = []
    for df_in, norm_sp, mx, mxs in [
        (train_df, train_norm, train_max, train_max_scaled),
        (val_df,   val_norm,   val_max,   val_max_scaled),
        (test_df,  test_norm,  test_max,  test_max_scaled),
    ]:
        df_out = df_in.copy()
        df_out["content"]            = [s.tolist() for s in norm_sp]
        df_out["max_amplitude"]      = mx.flatten()
        df_out["max_amplitude_scaled"] = mxs.flatten()
        for col in isotope_columns_to_normalize:
            if col in df_out.columns:
                df_out[col] = df_out[col] / 100.0
        df_out.attrs['global_min'] = float(global_min)
        df_out.attrs['global_max'] = float(global_max)
        normalized.append(df_out)

    return tuple(normalized), meta


# =============================================================================
# SAUVEGARDE
# =============================================================================

def save_datasets(datasets, filenames):
    for dataset, filename in zip(datasets, filenames):
        dataset.to_json(filename)
        print(f"Saved: {filename}  ({len(dataset)} rows)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    base_path = os.getcwd()
    print(f"Working directory: {base_path}")

    # ---- Colonnes isotopiques ----
    ISO_U  = ['234U',  '235U',  '236U',  '238U']

    N_CLUSTERS_U  = 3   # naturel / LEU / HEU

    # ---- 1. Chargement ----
    print("\n=== Loading spectra ===")
    df_raw = load_and_process_spectra(base_path)

    # ---- 2. Nettoyage ----
    print("\n=== Data cleaning ===")
    df_clean = clean_and_convert_data(df_raw)
    df_clean.to_json("dataset_before_normalization.json")
    create_histograms(df_clean)
    plot_isotope_dist_by_detector(df_clean, ISO_U, save_path="only_U/")


    # ---- 3. Filtrage ----
    print("\n=== Filtering ===")
    df_filtered = filter_and_normalize_data(df_clean)
    df_filtered.to_json("dataset_after_normalization.json")

    # ---- 4. Split ----
    print("\n=== Train/Val/Test split ===")
    train_dataset, val_dataset = stratified_train_val_split(
        df_filtered, train_ratio=0.8, random_seed=4
    )
    train_dataset.reset_index(inplace=True, drop=True)

    val_indices = np.arange(len(val_dataset))
    np.random.shuffle(val_indices)
    sp = len(val_dataset) // 2
    test_dataset = val_dataset.iloc[val_indices[sp:]].reset_index(drop=True)
    val_dataset  = val_dataset.iloc[val_indices[:sp]].reset_index(drop=True)

    # ---- 5. Structures communes ----
    clean_train = train_dataset[
        (train_dataset[ISO_U]  != -100).all(axis=1)
    ].copy()

    # Sous-ensembles U et Pu valides
    clean_U  = clean_train[(clean_train[ISO_U]  != -100).all(axis=1)].copy()

    # Clustering + bornes pour U
    cl_labels_U, _  = detect_isotope_clusters(
        clean_U, ISO_U, n_clusters=N_CLUSTERS_U, random_seed=42
    )
    bounds_U = compute_isotope_bounds_per_cluster(
        clean_U, cl_labels_U, ISO_U, N_CLUSTERS_U
    )
    ref_sp_U    = np.array([s for s in clean_U["content"].values])
    ref_comp_U  = clean_U[ISO_U].values.astype(float)
    dets_U      = clean_U["Detector"].values

    
    # ---- 6. Augmentation intra-cluster ----
    print("\n=== Augmentation intra-cluster U ===")
    aug_U1 = generate_linear_combination_spectra_constrained(
        clean_U, ISO_U, n_synthetic=500, min_spectra=2, max_spectra=4,
        random_seed=42, add_noise=True, n_clusters=N_CLUSTERS_U,
    )
    aug_U2 = generate_linear_combination_spectra_constrained(
        clean_U, ISO_U, n_synthetic=500, min_spectra=2, max_spectra=4,
        random_seed=43, add_noise=True, n_clusters=N_CLUSTERS_U,
    )


    # ---- 7. Augmentation inter-cluster U ----
    print("\n=== Augmentation inter-cluster U ===")
    inter_U = generate_intercluster_spectra(
        clean_U, cl_labels_U, bounds_U,
        ref_sp_U, ref_comp_U, dets_U, ISO_U,
        n_synthetic=300, cluster_pairs=None,
        add_noise=True, random_seed=44,
    )

    # ---- 8. Augmentation uniforme U (couvre les zones vides) ----
    print("\n=== Augmentation uniforme U ===")
    uniform_comps_U = generate_compositions_uniform(
        clean_U, ISO_U, n_synthetic=800, random_seed=45
    )
    uniform_U = generate_physically_consistent_spectra(
        uniform_comps_U, clean_U, ISO_U,
        ref_sp_U, ref_comp_U, dets_U,
        add_noise=True, random_seed=45,
    )
    # Ajout de Detector quanti au dataset uniforme
    dq_map = dict(clean_U[['Detector', 'Detector quanti']].drop_duplicates().values)
    uniform_U['Detector quanti'] = uniform_U['Detector'].map(dq_map)

    # ---- 9. Concaténation ----
    augmented_U  = pd.concat([aug_U1, aug_U2, inter_U, uniform_U], ignore_index=True)

    # Detector quanti pour tous
    dq_map_full = dict(
        train_dataset[['Detector', 'Detector quanti']].drop_duplicates().values
    )
    augmented_U['Detector quanti'] = (
        augmented_U['Detector quanti']
        .fillna(augmented_U['Detector'].map(dq_map_full))
    )

    print(f"\nDataset augmenté final : {len(augmented_U)} échantillons")
    print(f"  U  intra  : {len(aug_U1)+len(aug_U2)}")
    print(f"  U  inter  : {len(inter_U)}")
    print(f"  U  uniforme : {len(uniform_U)}")

    # ---- 10. Visualisation couverture ----
    plot_synthetic_vs_real_coverage(
        clean_U, augmented_U, ISO_U, save_path="only_U/"
    )

    # ---- 11. Sauvegarde avant normalisation ----
    save_datasets(
        (augmented_U, val_dataset, test_dataset),
        ["train_before_normalization_reduced_augmented.json",
         "val_before_normalization_reduced.json",
         "test_before_normalization_reduced.json"]
    )

    # ---- 12. Normalisation ----
    print("\n=== Normalisation ===")
    (train_norm, val_norm, test_norm), norm_meta = normalize_spectra(
        augmented_U, val_dataset, test_dataset,
        isotope_columns_to_normalize=ISO_U,
        save_path="only_U/"
    )
    create_histograms(train_norm, save_path="only_U/augmented_")
    plot_isotope_dist_by_detector(train_norm, ISO_U, save_path="only_U/augmented_")

    # ---- 13. Sauvegarde finale ----
    print("\n=== Final save ===")
    save_datasets(
        (train_norm, val_norm, test_norm),
        ["train.json", "val.json", "test.json"]
    )

    print(f"\nGlobal bounds: min={norm_meta['global_min']:.6f}, "
          f"max={norm_meta['global_max']:.6f}")
    print("\n=== Processing completed successfully ===")


if __name__ == "__main__":
    main()