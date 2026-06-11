"""
VAE Model Evaluation Script

This script evaluates a trained Variational Autoencoder (VAE) model with regression capabilities
for signal reconstruction and parameter estimation. It performs comprehensive analysis including:
- Model loading and evaluation
- Signal reconstruction quality assessment
- Latent space visualization
- Parameter regression performance evaluation
- Results export to CSV files
"""

import sys
import joblib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import argparse
from tqdm import tqdm
from torch.utils import data
from torchsummary import summary
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score, mean_squared_error
from torchview import draw_graph
import itertools

sys.path.append('../')
# Custom imports
from utils.functions import reconstruction_signal, latent_space_visualisation
from utils.preprocessing import Dataset_dataset
from models.model import VAE, Regressor

def interleave_arrays(a, b):
    """
    a, b : (N, 20)
    résultat : (N, 40) → [col0_a, col0_b, col1_a, col1_b, ...]
    """
    N, M = a.shape
    result = np.empty((N, 2 * M), dtype=a.dtype)
    result[:, 0::2] = a  # indices pairs   → colonnes de a
    result[:, 1::2] = b  # indices impairs → colonnes de b
    return result

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--date', default=None)
    parser.add_argument('--time', default=None)

    parser.add_argument('--draw_architecture', default=False, type=bool, help='Draw model architecture')
    parser.add_argument('--make_csv_file', default=False, type=bool, help='Generate CSV output files')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from JSON file."""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    config, _ = parser.parse_known_args()
    
    with open(f'{config_path}/config.txt', 'r') as f:
        config.__dict__ = json.load(f)
    
    return config


def set_default_config_values(config):
    """Set default values for missing configuration parameters."""
    defaults = {
        'last_act': "identity",
        'regression_with_mu': False,
        'train_with_regressor_with_only_mu': False,
        'model_version': 1
    }
    
    for key, value in defaults.items():
        if not hasattr(config, key):
            setattr(config, key, value)
    
    # Version-specific defaults
    if config.model_version == 2:
        config.norm_max = True


def load_datasets(config):
    """Load appropriate datasets based on configuration."""
    inputs_dim = 8192
    inputs_class = 5
    
    nb_classes = 1

    if hasattr(config, 'preprocessing_version'):
        config.preprocessing_version = config.preprocessing_version
    else:
        config.preprocessing_version = 2

    if config.preprocessing_version == 2:
        train_dataset = Dataset_dataset('../datasets/ESARDA/dataset Pu/X_train.json', config)
        train_without_augmentation_dataset = Dataset_dataset('../datasets/ESARDA/dataset Pu/X_train_without_augmentation.json', config)
        val_dataset = Dataset_dataset('../datasets/ESARDA/dataset Pu/X_val.json', config)
        test_dataset = Dataset_dataset('../datasets/ESARDA/dataset Pu/X_test.json', config)

    else:
        train_dataset = Dataset_dataset('../datasets/ESARDA/dataset Pu/train.json', config)
        train_without_augmentation_dataset = Dataset_dataset('../datasets/ESARDA/dataset Pu/train_without_augmentation.json', config)
        val_dataset = Dataset_dataset('../datasets/ESARDA/dataset Pu/val.json', config)
        test_dataset = Dataset_dataset('../datasets/ESARDA/dataset Pu/test.json', config)

   
   
    return (train_dataset, train_without_augmentation_dataset, val_dataset, test_dataset, 
            inputs_dim, inputs_class, nb_classes)


def load_models(config, load_path_vae, inputs_dim, inputs_class, device):
    """Load VAE and regressor models."""
    # Load VAE model
    model = VAE(config, device, inputs_dim, inputs_class).to(device)
    
    filename = "last_model.pt"
    weights = torch.load(load_path_vae + "/" + filename)
    
    if isinstance(weights, dict):
        model.load_state_dict(weights["weights"])
    else:
        model.load_state_dict(weights)
    
    model.eval()
           
    return model


def evaluate_model(model, data_loader, config, device):
    """Evaluate model on given dataset."""
    truth_data = []
    reconstruction_data = []
    mu_data = []
    sigma_data = []
    norm_param_pred_data = []
    norm_param_truth_data = []
    error_reconstruction = []
    condition_list = []

    with torch.no_grad():
        for batch, inputs in enumerate(data_loader):
            X, condition, y = inputs
            X, condition, y = X.to(device), condition.to(device), y.to(device)
                    
            # Forward pass through VAE
            reconstruction, mu, sigma, norm_param_pred = model.exact_reconstruction(X,condition= condition)
           
            
            # Store results
            truth_data.append(X.cpu().detach().numpy())
            reconstruction_data.append(reconstruction.cpu().detach().numpy())
            mu_data.append(mu.cpu().detach().numpy())
            sigma_data.append(torch.exp(0.5 * sigma).cpu().detach().numpy())
            norm_param_truth_data.append(y.cpu().detach().numpy())
            norm_param_pred_data.append(norm_param_pred.cpu().detach().numpy())
            
            # Calculate reconstruction error (MSE)
            reconstruction_error = ((X.cpu().detach().numpy() - reconstruction.cpu().detach().numpy()) ** 2).mean(axis=1)
            error_reconstruction.append(reconstruction_error)
            condition_list.append(condition.cpu().detach().numpy())

    # Concatenate all batches
    results = {
        'truth': np.concatenate(truth_data, axis=0),
        'reconstruction': np.concatenate(reconstruction_data, axis=0),
        'mu': np.concatenate(mu_data, axis=0),
        'sigma': np.concatenate(sigma_data, axis=0),
        'norm_param_truth': np.concatenate(norm_param_truth_data, axis=0),
        'norm_param_pred': np.concatenate(norm_param_pred_data, axis=0),
        'error_reconstruction': np.concatenate(error_reconstruction, axis=0),
        'condition': np.concatenate(condition_list, axis=0),
        'condition_name' : data_loader.dataset.condition_name, 
        'detector_list' : data_loader.dataset.detector_list
    }
    
    return results


def create_visualizations(results_train, results_train_without_augmentation, results_val, results_test, config, path):
    """Create various visualizations for model evaluation."""
    # Combine all results
    all_mu = np.concatenate([results_train['mu'], results_train_without_augmentation['mu'], results_val['mu'], results_test['mu']], axis=0)
    all_condition = np.concatenate([results_train['condition'], results_train_without_augmentation['condition'], results_val['condition'], results_test['condition']], axis=0)
    all_pred = np.concatenate([results_train['norm_param_truth'], results_train_without_augmentation['norm_param_truth'], results_val['norm_param_truth'], results_test['norm_param_truth']], axis=0)

    # Create t-SNE visualization of latent space
    print("Creating t-SNE visualization...")
    tsne = TSNE(n_components=2, random_state=42)
    latent_space_2d = tsne.fit_transform(all_mu)
    
    # Visualize latent space with different conditions
    
    output_names = ['Pu-238/Pu', 'Pu-239/Pu', 'Pu-240/Pu', 'Pu-241/Pu', 'Pu-242/Pu']
    output_types = ["quanti", "quanti", "quanti", "quanti", "quanti", "quanti"]
    labels_classe = []
    for n_output, (output_name, output_type) in enumerate(zip(output_names, output_types)):
        
        new_output_name = output_name.replace("/", "-")

        labels_classe = np.unique(all_pred[:,n_output])
        latent_space_visualisation(latent_space_2d, all_pred[:,n_output], labels_classe, f"{path}/LS visualisation hue {new_output_name}", viridis = True, type = output_type)
    
    condition_names = results_train["condition_name"]
    condition_types = ["quali", "quanti", "quanti"]
    detector_list = results_train["detector_list"]


    labels_classe = []
    for n_condition, (condition_name, condition_type) in enumerate(zip(condition_names, condition_types)):
        
        new_condition_name = condition_name.replace("/", "-")

        
        if condition_type == "quali":

            labels_classe = detector_list
        else:
            labels_classe = np.unique(all_condition[:,n_condition])
        
        latent_space_visualisation(latent_space_2d, all_condition[:,n_condition], labels_classe, f"{path}/LS visualisation hue {new_condition_name}", viridis = True, type = condition_type)
    
    # Visualize by dataset split
    dataset_labels = np.array([0] * len(results_train['mu']) + 
                             [1] * len(results_train_without_augmentation['mu']) + 
                             [2] * len(results_val['mu']) + 
                             [3] * len(results_test['mu']))
    latent_space_visualisation(
        latent_space_2d, 
        dataset_labels, 
        ["train", "train_without_augmentation", "val", "test"], 
        f"{path}/LS_visualisation_train_val_test", 
        type="quali"
    )
    
    return latent_space_2d


def plot_r2_scores(results_train, results_train_without_augmentation, results_val, results_test, config, path):
    """Plot R² scores for parameter prediction."""
    
    datasets = [
        (results_train, "train"),
        (results_train_without_augmentation, "train_without_augmentation"),
        (results_val, "val"), 
        (results_test, "test")
    ]
    
    for results, dataset_name in datasets:        
        for i, isotop in  enumerate(['Pu-238/Pu', 'Pu-239/Pu', 'Pu-240/Pu', 'Pu-241/Pu', 'Pu-242/Pu']):
        

            isotop_name = isotop.replace("/", "-")
            # Single output case
            
            r2_score_val = r2_score(results['norm_param_truth'][:,i], results['norm_param_pred'][:,i])
            fig = plt.figure(figsize=(8, 5))

            plt.scatter(results['norm_param_truth'][:,i], results['norm_param_pred'][:,i])
            min_ = min(np.min(results['norm_param_truth'][:,i]), np.min(results['norm_param_pred'][:,i]))
            max_ = max(np.max(results['norm_param_truth'][:,i]), np.max(results['norm_param_pred'][:,i]))

            plt.plot([min_, max_], [min_, max_], color='r')
            plt.xlabel("Truth")
            plt.ylabel("Prediction")
            plt.title(f"R²: {r2_score_val:.4f}")
        
            plt.tight_layout()
            fig.savefig(f"{path}/R2_{isotop_name}_{dataset_name}.png", dpi=300, bbox_inches='tight')
            plt.close()

def save_results(results_train, results_train_without_augmentation, results_val, results_test, config, detector_dict, path):
    """Save evaluation results and statistics."""
    

    iso_names = ['Pu-238/Pu (truth)', 'Pu-239/Pu (truth)', 'Pu-240/Pu (truth)', 'Pu-241/Pu (truth)', 'Pu-242/Pu(truth)']
    iso_names2 = ['Pu-238/Pu (pred)', 'Pu-239/Pu (pred)', 'Pu-240/Pu (pred)', 'Pu-241/Pu (pred)', 'Pu-242/Pu (pred)']
    iso_names_mix = list(itertools.chain.from_iterable(zip(iso_names,iso_names2)))
    norm_param_train = interleave_arrays(results_train["norm_param_truth"], results_train["norm_param_pred"])
    norm_param_train_without_augmentation = interleave_arrays(results_train_without_augmentation["norm_param_truth"], results_train_without_augmentation["norm_param_pred"])
    norm_param_val   = interleave_arrays(results_val["norm_param_truth"],   results_val["norm_param_pred"])
    norm_param_test  = interleave_arrays(results_test["norm_param_truth"],  results_test["norm_param_pred"])
    # base_cols = results_train["condition_name"] + iso_names_mix

     
    results_train["Detector name"] = np.array([detector_dict.get(int(key), f"Unknown ({int(key)})") for key in results_train["condition"][:, 0]])[:, np.newaxis]
    results_train_without_augmentation["Detector name"] = np.array([detector_dict.get(int(key), f"Unknown ({int(key)})") for key in results_train_without_augmentation["condition"][:, 0]])[:, np.newaxis]
    results_val["Detector name"] = np.array([detector_dict.get(int(key), f"Unknown ({int(key)})") for key in results_val["condition"][:, 0]])[:, np.newaxis]
    results_test["Detector name"] = np.array([detector_dict.get(int(key), f"Unknown ({int(key)})") for key in results_test["condition"][:, 0]])[:, np.newaxis]
    base_cols = ["File name", "Detector", "Detector name", "max_amplitude", "ENER_FIT"] + iso_names_mix

    base_cols = ["File name", "Detector name"] + results_train["condition_name"] + iso_names_mix

    if "tsne" in results_train:
        tsne_cols = ["tsne_0", "tsne_1"]
        
        data_train = np.hstack((results_train["File name"], results_train["Detector name"], results_train["condition"], norm_param_train, results_train["tsne"]))
        data_train_without_augmentation = np.hstack((results_train_without_augmentation["File name"], results_train_without_augmentation["Detector name"], results_train_without_augmentation["condition"], norm_param_train_without_augmentation, results_train_without_augmentation["tsne"]))
        data_val = np.hstack((results_val["File name"], results_val["Detector name"], results_val["condition"], norm_param_val, results_val["tsne"]))
        data_test = np.hstack((results_test["File name"], results_test["Detector name"], results_test["condition"], norm_param_test, results_test["tsne"]))
        
        cols = base_cols + tsne_cols
    elif "mu" in results_train:
        latent_dim = results_train["mu"].shape[1]
        latent_cols = [f"latent_dim_{i}" for i in range(latent_dim)]
        
        data_train = np.hstack((results_train["File name"], results_train["Detector name"], results_train["condition"], norm_param_train, results_train["mu"]))
        data_train_without_augmentation = np.hstack((results_train_without_augmentation["File name"], results_train_without_augmentation["Detector name"], results_train_without_augmentation["condition"], norm_param_train_without_augmentation, results_train_without_augmentation["mu"]))
        data_val = np.hstack((results_val["File name"], results_val["Detector name"], results_val["condition"], norm_param_val, results_val["mu"]))
        data_test = np.hstack((results_test["File name"], results_test["Detector name"], results_test["condition"], norm_param_test, results_test["mu"]))
        
        cols = base_cols + latent_cols
    else:
        data_train = np.hstack((results_train["File name"], results_train["Detector name"], results_train["condition"], norm_param_train))
        data_train_without_augmentation = np.hstack((results_train_without_augmentation["File name"], results_train_without_augmentation["Detector name"], results_train_without_augmentation["condition"], norm_param_train_without_augmentation))
        data_val = np.hstack((results_val["File name"], results_val["Detector name"], results_val["condition"], norm_param_val))
        data_test = np.hstack((results_test["File name"], results_test["Detector name"], results_test["condition"], norm_param_test))
        
        cols = base_cols

    final_df_train = pd.DataFrame(data_train, columns=cols)
    final_df_train_without_augmentation = pd.DataFrame(data_train_without_augmentation, columns=cols)
    final_df_val = pd.DataFrame(data_val, columns=cols)
    final_df_test = pd.DataFrame(data_test, columns=cols)

    final_df_train["dataset"] = ["train"]*len(final_df_train)
    final_df_train_without_augmentation["dataset"] = ["train"]*len(final_df_train_without_augmentation)

    final_df_train[f"reconstruction_error"] = np.round(results_train["error_reconstruction"], decimals=4)
    final_df_train_without_augmentation[f"reconstruction_error"] = np.round(results_train_without_augmentation["error_reconstruction"], decimals=4)
    print(len(final_df_train))
    print(len(final_df_train_without_augmentation))

    final_df_val["dataset"] = ["val"]*len(final_df_val)
    final_df_val[f"reconstruction_error"] = np.round(results_val["error_reconstruction"], decimals=4)

    print(len(final_df_val))

    final_df_test["dataset"] = ["test"]*len(final_df_test)
    final_df_test[f"reconstruction_error"] = np.round(results_test["error_reconstruction"], decimals=4)

    
    print(len(final_df_test))
    print(final_df_train.shape, final_df_val.shape, final_df_test.shape)

    print(final_df_train.columns)
    print(final_df_val.columns)
    print(final_df_test.columns)

    final_df = pd.concat([final_df_train, final_df_train_without_augmentation, final_df_val, final_df_test])

    final_df.reset_index(inplace = True, drop=True)

    print(final_df)

    final_df.to_csv(f"{path}/results_global.csv")


def main():
    """Main evaluation function."""
    # Clear GPU cache
    torch.cuda.empty_cache()
    
    # Parse arguments
    config_test = parse_arguments()
    
    print('=' * 50)
    print(f'Starting VAE Model Evaluation')
    print('=' * 50)
    
    if config_test.time is None or config_test.date is None:
        dataset_name = "ESARDA"
    else:
        dataset_name = f"{config_test.date}/{config_test.time}"

    # Load VAE configuration
    load_path_vae = f"../weights/{dataset_name}"
    config = load_config(load_path_vae)
    set_default_config_values(config)
  
    # Load datasets
    (train_dataset, train_without_augmentation_dataset, val_dataset, test_dataset, 
     inputs_dim, inputs_class, nb_classes) = load_datasets(config)
    
    # Create data loaders
    train_loader = data.DataLoader(train_dataset, batch_size=config.batch_size, num_workers=config.num_workers)
    train_without_augmentation_loader = data.DataLoader(train_without_augmentation_dataset, batch_size=config.batch_size, num_workers=config.num_workers)
    val_loader = data.DataLoader(val_dataset, batch_size=config.batch_size, num_workers=config.num_workers)
    test_loader = data.DataLoader(test_dataset, batch_size=config.batch_size, num_workers=config.num_workers)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')
    
    # Load models
    model = load_models(config, load_path_vae, inputs_dim, inputs_class, device)
    
    # Print model summary
    print(f"Input class dimension: {inputs_class}")
    print(summary(model, (inputs_dim,)))
    
    # Create output directory
    output_path = f"../outputs/{dataset_name}"
    os.makedirs(output_path, exist_ok=True)
    
    # Draw architecture if requested
    if config_test.draw_architecture:
        model_graph = draw_graph(
            model, 
            input_size=((4, inputs_dim), (4, inputs_class)), 
            save_graph=True, 
            filename=f"{output_path}/architecture", 
            expand_nested=True
        )
    
    print("Evaluating model...")
    
    # Evaluate model on all datasets
    results_train = evaluate_model(model, train_loader, config, device)
    results_train_without_augmentation = evaluate_model(model, train_without_augmentation_loader, config, device)
    results_val = evaluate_model(model, val_loader, config, device)
    results_test = evaluate_model(model, test_loader, config, device)
    
    # Create visualizations
    detector_dict = train_dataset.detector_dict
    latent_space_2d = create_visualizations(results_train, results_train_without_augmentation, results_val, results_test, config, output_path)
    
    len_train = len(results_train['mu'])
    len_train_without_augmentation = len(results_train_without_augmentation['mu'])
    len_val = len(results_val['mu'])
    
    results_train['tsne'] = latent_space_2d[:len_train]
    results_train_without_augmentation['tsne'] = latent_space_2d[len_train:len_train+len_train_without_augmentation]
    results_val['tsne'] = latent_space_2d[len_train+len_train_without_augmentation:len_train+len_train_without_augmentation+len_val]
    results_test['tsne'] = latent_space_2d[len_train+len_train_without_augmentation+len_val:]

    results_train['File name'] = train_dataset.file_name
    results_train_without_augmentation['File name'] = train_without_augmentation_dataset.file_name
    results_val['File name'] = val_dataset.file_name
    results_test['File name'] = test_dataset.file_name

    save_results(results_train, results_train_without_augmentation, results_val, results_test, config, detector_dict, output_path)
    
    # Plot R² scores
    plot_r2_scores(results_train, results_train_without_augmentation, results_val, results_test, config, output_path)
    
    # Plot some reconstruction examples
    print("Creating reconstruction plots...")
    random_indices_train = np.random.choice(len(results_train['reconstruction']), size=5, replace=False)
    random_indices_train_without_augmentation = np.random.choice(len(results_train_without_augmentation['reconstruction']), size=5, replace=False)
    random_indices_val = np.random.choice(len(results_val['reconstruction']), size=5, replace=False)
    random_indices_test = np.random.choice(len(results_test['reconstruction']), size=5, replace=False)

    reconstruction_signal(
        results_train['reconstruction'][random_indices_train],
        results_train['truth'][random_indices_train],
        f"{output_path}/reconstruction_plots_train_normalized_signal",
        isotopic_preds=results_train['norm_param_pred'][random_indices_train],
        isotopic_truths=results_train['norm_param_truth'][random_indices_train]
    )

    reconstruction_signal(
        results_train_without_augmentation['reconstruction'][random_indices_train_without_augmentation],
        results_train_without_augmentation['truth'][random_indices_train_without_augmentation],
        f"{output_path}/reconstruction_plots_train_without_augmentation_normalized_signal",
        isotopic_preds=results_train_without_augmentation['norm_param_pred'][random_indices_train_without_augmentation],
        isotopic_truths=results_train_without_augmentation['norm_param_truth'][random_indices_train_without_augmentation]
    )

    reconstruction_signal(
        results_val['reconstruction'][random_indices_val],
        results_val['truth'][random_indices_val],
        f"{output_path}/reconstruction_plots_val_normalized_signal",
        isotopic_preds=results_val['norm_param_pred'][random_indices_val],
        isotopic_truths=results_val['norm_param_truth'][random_indices_val]
    )

    reconstruction_signal(
        results_test['reconstruction'][random_indices_test],
        results_test['truth'][random_indices_test],
        f"{output_path}/reconstruction_plots_test_normalized_signal",
        isotopic_preds=results_test['norm_param_pred'][random_indices_test],
        isotopic_truths=results_test['norm_param_truth'][random_indices_test]
    )
    
    r2_score_test = []
    for i in range(5):
        r2_score_test.append(r2_score(results_test['norm_param_truth'][:,i], results_test['norm_param_pred'][:,i]))
    
    r2_score_test = np.mean(np.array(r2_score_test))

    # Write summary statistics
    with open(f"{output_path}/evaluation_summary.txt", "w") as f:
        f.write("VAE Model Evaluation Summary\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"Latent dimension: {config.latent_dim}\n")
       
        f.write("Reconstruction Errors (MSE on normalized data):\n")
        f.write(f"  Train: {np.mean(results_train['error_reconstruction']):.6f}\n")
        f.write(f"  Train without augmentation: {np.mean(results_train_without_augmentation['error_reconstruction']):.6f}\n")
        f.write(f"  Val:   {np.mean(results_val['error_reconstruction']):.6f}\n")
        f.write(f"  Test:  {np.mean(results_test['error_reconstruction']):.6f}\n\n")
        f.write(f" R2  Test:  {r2_score_test}\n\n")

    print("=" * 50)
    print("Evaluation completed successfully!")
    print(f"Results saved to: {output_path}")
    print("=" * 50)


if __name__ == '__main__':
    main()