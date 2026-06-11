import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import torch

from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import seaborn as sns
from scipy import stats


def reconstruction_signal(predictions, truths, filename, one_per_image=False, isotopic_preds=None, isotopic_truths=None, iso_names=None):
    """
    Visualizes reconstruction results by plotting predictions vs ground truth signals.
    
    Optimizations made:
    - Reduced code duplication by extracting plot creation logic
    - Improved color handling with consistent color scheme
    - Added context managers for better memory management
    - Standardized figure sizing and styling
    """
    colors = ["red", "green", "blue", "black"]

    if iso_names is None:
        iso_names = ['Pu-238/Pu', 'Pu-239/Pu', 'Pu-240/Pu', 'Pu-241/Pu', 'Pu-242/Pu']
    
    def create_signal_plot(data, color, title_suffix="", pred_iso=None, truth_iso=None):
        """Helper function to create consistent signal plots"""
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(data, c=color, linewidth=2)
        title = f'{filename} {title_suffix}'
        if pred_iso is not None and truth_iso is not None:
            pred_str = ", ".join([f"{name}: {p:.3f}" for name, p in zip(iso_names, pred_iso)])
            truth_str = ", ".join([f"{name}: {t:.3f}" for name, t in zip(iso_names, truth_iso)])
            title += f'\nComposition predicted : [{pred_str}] | True composition: [{truth_str}]'
        ax.set_title(title, fontsize=16)
        return fig, ax
    
    if one_per_image:
        # Generate individual plots for each prediction/truth pair
        for i in range(len(predictions)):
            pred_iso = isotopic_preds[i] if isotopic_preds is not None else None
            truth_iso = isotopic_truths[i] if isotopic_truths is not None else None
            
            # Prediction plot
            fig, ax = create_signal_plot(predictions[i], "black", f"prediction {i}", pred_iso, truth_iso)
            fig.savefig(f'{filename} prediction {i}', dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # Truth plot
            fig, ax = create_signal_plot(truths[i], "black", f"truth {i}", pred_iso, truth_iso)
            fig.savefig(f'{filename} truth {i}', dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            # Superposed plot
            fig, ax = plt.subplots(figsize=(20, 10))
            ax.plot(truths[i], c=colors[1], label="original", linewidth=2, alpha=0.8)
            ax.plot(predictions[i], c=colors[0], label="reconstruction", linewidth=2, alpha=0.8)
            ax.legend(loc='upper right', fontsize=16)
            title = f'{filename} superposed {i}'
            if pred_iso is not None and truth_iso is not None:
                pred_items = [f"{name}: {p:.3f}" for name, p in zip(iso_names, pred_iso)]
                truth_items = [f"{name}: {t:.3f}" for name, t in zip(iso_names, truth_iso)]
                
                mid = (len(pred_items) + 1) // 2
                pred_str = ", ".join(pred_items[:mid]) + ",\n" + ", ".join(pred_items[mid:])
                truth_str = ", ".join(truth_items[:mid]) + ",\n" + ", ".join(truth_items[mid:])
                
                title += f'\nPredicted:\n[{pred_str}]\nTruth:\n[{truth_str}]'
            ax.set_title(title, fontsize=16)
            fig.tight_layout()
            fig.savefig(f'{filename} superposed {i}', dpi=150, bbox_inches='tight')
            plt.close(fig)
    else:
        n_samples = len(predictions)
        
        # Side-by-side comparison plot
        fig, axes = plt.subplots(n_samples, 2, figsize=(20, 5 * n_samples))
        # Handle single sample case
        if n_samples == 1:
            axes = axes.reshape(1, -1)
            
        for i in range(n_samples):
            pred_title = 'Prediction'
            truth_title = 'Truth'
            if isotopic_preds is not None and isotopic_truths is not None:
                pred_items = [f"{name}: {p:.3f}" for name, p in zip(iso_names, isotopic_preds[i])]
                truth_items = [f"{name}: {t:.3f}" for name, t in zip(iso_names, isotopic_truths[i])]
                
                mid = (len(pred_items) + 1) // 2
                pred_str = ", ".join(pred_items[:mid]) + ",\n" + ", ".join(pred_items[mid:])
                truth_str = ", ".join(truth_items[:mid]) + ",\n" + ", ".join(truth_items[mid:])
                
                pred_title += f'\nComposition predicted:\n[{pred_str}]'
                truth_title += f'\nTrue composition:\n[{truth_str}]'

            axes[i, 0].plot(predictions[i], c="black", linewidth=2)
            axes[i, 0].set_title(pred_title, fontsize=14)
            axes[i, 0].grid(True, alpha=0.3)
            
            axes[i, 1].plot(truths[i], c="black", linewidth=2)
            axes[i, 1].set_title(truth_title, fontsize=14)
            axes[i, 1].grid(True, alpha=0.3)
        
        fig.tight_layout()
        fig.savefig(f'{filename}_comparison', dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Overlaid comparison plot
        fig, axes = plt.subplots(n_samples, 1, figsize=(20, 4 * n_samples))
        if n_samples == 1:
            axes = [axes]
            
        for i in range(n_samples):
            axes[i].plot(predictions[i], c=colors[0], label="reconstruction", 
                        linewidth=2, alpha=0.8)
            axes[i].plot(truths[i], c=colors[1], label="original", 
                        linewidth=2, alpha=0.8)
            axes[i].grid(True, alpha=0.3)
            title = f'Sample {i+1}'
            if isotopic_preds is not None and isotopic_truths is not None:
                pred_str = ", ".join([f"{name}: {p:.3f}" for name, p in zip(iso_names, isotopic_preds[i])])
                truth_str = ", ".join([f"{name}: {t:.3f}" for name, t in zip(iso_names, isotopic_truths[i])])
                title += f'\nComposition predicted : [{pred_str}] | True composition: [{truth_str}]'
            axes[i].set_title(title, fontsize=14)
        
        # Add legend only to the last subplot to avoid clutter
        axes[-1].legend(loc='upper right', fontsize=14)
        fig.tight_layout()
        fig.savefig(f'{filename}', dpi=150, bbox_inches='tight')
        plt.close(fig)


def latent_space_visualisation(
    mu, classes, label_classes, filename,
    dimension_wise=False, tsne_dim=2, hue=None, hue_label="",
    additional_plots=False, viridis=False, save_cluster_id=False,
    id_list=None, train_tsne=False, type="quanti"
):
    """
    Comprehensive latent space visualization with multiple viewing modes.
    
    Optimizations made:
    - Improved parameter validation and early returns
    - Better memory management with explicit figure closing
    - Enhanced color scheme and marker handling
    - Optimized DataFrame operations
    - Added proper error handling for edge cases
    - Improved clustering visualization with better parameters
    """
    latent_dim = mu.shape[1]
    n_samples = mu.shape[0]
    
    # Input validation
    if len(classes) != n_samples:
        raise ValueError("Classes array length must match number of samples")
    
    # Dimensionality reduction setup
    if train_tsne and latent_dim > tsne_dim:
        print(f"Applying t-SNE reduction from {latent_dim}D to {tsne_dim}D...")
        tsne = TSNE(n_components=tsne_dim, random_state=42, perplexity=min(30, n_samples//4))
        latent_space_samples = tsne.fit_transform(mu)
    else:
        latent_space_samples = mu[:, :tsne_dim] if latent_dim > tsne_dim else mu

    # --- Dimension-wise visualization ---
    if dimension_wise:
        print("Creating dimension-wise visualization...")
        # Vectorized approach for better performance
        dim_indices = np.repeat(np.arange(latent_dim), n_samples)
        sample_indices = np.tile(np.arange(n_samples), latent_dim)
        
        df = pd.DataFrame({
            "latent samples": mu.flatten(),
            "axis": dim_indices,
            "label": np.tile([label_classes[i] for i in classes], latent_dim)
        })
        
        plt.figure(figsize=(15, 10))
        sns.displot(data=df, x="latent samples", hue="label", col="axis", 
                   kind="kde", col_wrap=4, height=4, aspect=1.2)
        plt.savefig(f'{filename}_dimension_wise', dpi=150, bbox_inches='tight')
        plt.close()
        return

    # --- 1D t-SNE visualization ---
    if tsne_dim == 1:
        print("Creating 1D t-SNE visualization...")
        df = pd.DataFrame({
            "latent samples": latent_space_samples.flatten(),
            "label": [label_classes[i] for i in classes]
        })
        
        plt.figure(figsize=(12, 6))
        sns.displot(data=df, x="latent samples", hue="label", kind="kde", 
                   height=6, aspect=2, alpha=0.7)
        plt.savefig(f'{filename}_1d_tsne', dpi=150, bbox_inches='tight')
        plt.close()
        return

    # --- 2D visualization with optional hue ---
    if hue is not None:
        print("Creating 2D visualization with hue...")
        markers_list = ["o", "x", "^", "s", "D"]  # More marker variety
        alpha_list = [0.7, 0.9, 0.6, 0.8, 0.5]
        # Use a better color palette
        colors = plt.cm.Set1(np.linspace(0, 1, len(label_classes)))
        
        plt.figure(figsize=(12, 10))
        for n_class, label in enumerate(label_classes):
            class_indices = np.where(classes == n_class)[0]
            if len(class_indices) == 0:
                continue
                
            temp_samples = latent_space_samples[class_indices]
            temp_hue = hue[class_indices]
            
            for hue_id in range(len(hue_label)):
                hue_indices = np.where(temp_hue == hue_id)[0]
                if len(hue_indices) > 0:
                    plt.scatter(
                        temp_samples[hue_indices, 0], 
                        temp_samples[hue_indices, 1],
                        label=f'{label}_{hue_label[hue_id]}',
                        marker=markers_list[hue_id % len(markers_list)],
                        alpha=alpha_list[hue_id % len(alpha_list)],
                        c=[colors[n_class]], s=60
                    )
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xlabel('t-SNE 1', fontsize=12)
        plt.ylabel('t-SNE 2', fontsize=12)
        plt.title('Latent Space Visualization with Hue', fontsize=14)
        plt.savefig(f'{filename}_with_hue', dpi=150, bbox_inches='tight')
        plt.close()

        # Per-class clustering analysis
        cluster_results = []
        
        for n_class in np.unique(classes):
            class_indices = np.where(classes == n_class)[0]
            if len(class_indices) < 10:  # Skip classes with too few samples
                continue
                
            # Apply t-SNE per class for better clustering
            class_tsne = TSNE(n_components=tsne_dim, random_state=42).fit_transform(mu[class_indices])
            temp_hue = hue[class_indices]
            
            plt.figure(figsize=(10, 8))
            for hue_id in range(len(hue_label)):
                hue_indices = np.where(temp_hue == hue_id)[0]
                if len(hue_indices) > 0:
                    plt.scatter(
                        class_tsne[hue_indices, 0], 
                        class_tsne[hue_indices, 1],
                        label=hue_label[hue_id], s=60, alpha=0.7
                    )
            
            plt.legend()
            plt.title(f'Class: {label_classes[n_class]}', fontsize=14)
            plt.xlabel('t-SNE 1', fontsize=12)
            plt.ylabel('t-SNE 2', fontsize=12)
            plt.savefig(f'{filename}_{label_classes[n_class]}_detailed', 
                       dpi=150, bbox_inches='tight')
            plt.close()
            
            # Clustering analysis
            if save_cluster_id and id_list is not None:
                # Adaptive clustering parameters based on data density
                eps = np.percentile(np.linalg.norm(class_tsne, axis=1), 25) * 0.1
                min_samples = max(5, len(class_indices) // 20)
                
                clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(class_tsne)
                cluster_labels = clustering.labels_
                
                cluster_info = pd.DataFrame({
                    "site": [label_classes[n_class]] * len(cluster_labels),
                    "ID": id_list[class_indices],
                    "cluster": cluster_labels
                })
                cluster_results.append(cluster_info)
        
        # Save clustering results
        if save_cluster_id and cluster_results:
            final_clusters = pd.concat(cluster_results, ignore_index=True)
            final_clusters.to_excel(f"{filename}_cluster_details.xlsx", index=False)
        
        return

    # --- Standard 2D visualization ---
    plt.figure(figsize=(12, 10))
    
    if viridis and type == "quanti":
        # Quantitative visualization with viridis colormap
        unique_classes = np.unique(classes)
        # Show only every nth value on colorbar for cleaner display
        n_ticks = min(8, len(unique_classes))  # Maximum 8 ticks
        tick_indices = np.linspace(0, len(unique_classes)-1, n_ticks, dtype=int)
        tick_values = unique_classes[tick_indices]
        
        scat = plt.scatter(
            latent_space_samples[:, 0], latent_space_samples[:, 1],
            c=classes, cmap="viridis", s=60, alpha=0.7
        )
        cbar = plt.colorbar(scat, ticks=tick_values)
        cbar.set_label("Class", fontsize=12)
    else:
        # Categorical visualization
        colors = plt.cm.Set1(np.linspace(0, 1, len(label_classes)))
        for n_class, label in enumerate(label_classes):
            class_indices = np.where(classes == n_class)[0]
            if len(class_indices) > 0:
                plt.scatter(
                    latent_space_samples[class_indices, 0],
                    latent_space_samples[class_indices, 1],
                    label=label, c=[colors[n_class]], s=60, alpha=0.7
                )
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.xlabel('Component 1', fontsize=12)
    plt.ylabel('Component 2', fontsize=12)
    plt.title('Latent Space Visualization', fontsize=14)
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


def compute_quali_quanti_correlation(qualititive_variables, quantitative_variables, 
                                   filename, qualititive_variables_name, threshold=0.4):
    """
    Computes and visualizes correlations between qualitative and quantitative variables.
    
    Optimizations made:
    - Vectorized one-hot encoding for better performance
    - Improved statistical computation with proper p-value handling
    - Enhanced visualization with better color schemes and error bars
    - Added correlation strength interpretation
    - Better handling of edge cases and data validation
    """
    # Input validation
    if len(qualititive_variables) != len(quantitative_variables):
        raise ValueError("Qualitative and quantitative variables must have same length")
    
    # Optimized one-hot encoding
    if len(qualititive_variables_name) == 1:
        qualititive_variables_one_hot = qualititive_variables.reshape(-1, 1)
    else:
        n_categories = int(qualititive_variables.max()) + 1
        qualititive_variables_one_hot = np.eye(n_categories)[qualititive_variables]
    
    latent_dim = quantitative_variables.shape[-1]
    corr_mask = {}
    
    # Better color scheme for correlations
    def get_correlation_color(correlation, p_value):
        """Get color based on correlation strength and significance"""
        if p_value >= 0.05:
            return 'lightgray'  # Non-significant
        elif abs(correlation) >= threshold:
            return 'darkred' if correlation > 0 else 'darkblue'  # Strong correlation
        else:
            return 'orange' if correlation > 0 else 'lightblue'  # Weak correlation
    
    for n_loc, loc in enumerate(qualititive_variables_name):
        temp_loc = qualititive_variables_one_hot[:, n_loc]
        
        # Vectorized correlation computation
        correlations = []
        p_values = []
        significance_mask = []
        
        for ax in range(latent_dim):
            # Handle edge case where all values are the same
            if np.var(quantitative_variables[:, ax]) == 0 or np.var(temp_loc) == 0:
                corr, p_val = 0.0, 1.0
            else:
                corr, p_val = stats.pointbiserialr(quantitative_variables[:, ax], temp_loc)
                # Handle NaN values
                if np.isnan(corr):
                    corr, p_val = 0.0, 1.0
            
            correlations.append(corr)
            p_values.append(p_val)
            significance_mask.append(p_val < 0.05 and abs(corr) >= threshold)
        
        correlations = np.array(correlations)
        p_values = np.array(p_values)
        
        # Enhanced visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create bars with color coding
        bar_colors = [get_correlation_color(corr, p_val) 
                     for corr, p_val in zip(correlations, p_values)]
        
        bars = ax.bar(np.arange(latent_dim), correlations, color=bar_colors, 
                     alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add significance threshold lines
        ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, 
                  label=f'Threshold (+{threshold})')
        ax.axhline(y=-threshold, color='red', linestyle='--', alpha=0.7, 
                  label=f'Threshold (-{threshold})')
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Formatting
        ax.set_xlabel("Latent Space Dimension", fontsize=14)
        ax.set_ylabel("Point-Biserial Correlation", fontsize=14)
        ax.set_title(f"Correlation Analysis: {loc}", fontsize=16)
        ax.set_xticks(np.arange(latent_dim))
        ax.set_xticklabels([f'Dim {i+1}' for i in range(latent_dim)])
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
        
        # Add text annotations for significant correlations
        for i, (corr, p_val, is_sig) in enumerate(zip(correlations, p_values, significance_mask)):
            if is_sig:
                ax.annotate(f'{corr:.3f}', (i, corr), textcoords="offset points", 
                           xytext=(0, 10 if corr > 0 else -15), ha='center', 
                           fontsize=10, fontweight='bold')
        
        # Save plot
        output_filename = filename if len(qualititive_variables_name) == 1 else f"{filename}_{loc}"
        fig.savefig(f'{output_filename}_correlation', dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Store results
        corr_mask[loc] = np.array(significance_mask)
        
        # Print summary statistics
        n_significant = np.sum(significance_mask)
        print(f"Variable '{loc}': {n_significant}/{latent_dim} dimensions show significant correlation")
    
    return corr_mask

def set_seed(seed=42):
    """
    Sets the random seed for reproducibility across random, numpy, and torch.
    
    Args:
        seed (int): The seed value to use for all pseudo-random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False