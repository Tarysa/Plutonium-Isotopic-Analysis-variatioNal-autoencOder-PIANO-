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

sys.path.append('../')
# Custom imports
from utils.functions import reconstruction_signal, latent_space_visualisation
from utils.preprocessing_IDB import Dataset_dataset
from models.model_IDB import VAE, Regressor


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
    inputs_dim = 4096
    inputs_class = 3
    
    nb_classes = 1

    train_dataset = Dataset_dataset('../datasets/IDB/train.json', config)
    val_dataset = Dataset_dataset('../datasets/IDB/val.json', config)
    test_dataset = Dataset_dataset('../datasets/IDB/test.json', config)

   
    return (train_dataset, val_dataset, test_dataset, 
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
    norm_param_truth_u = []
    norm_param_truth_pu = []
    norm_param_pred_u_list = []
    norm_param_pred_pu_list = []

    mask_u_list = []
    mask_pu_list = []

    error_reconstruction = []
    
    with torch.no_grad():
        for batch, inputs in enumerate(data_loader):
            X, u_proportion, pu_proportion, condition = inputs
            X,  u_proportion, pu_proportion, condition = X.to(device), u_proportion.to(device), pu_proportion.to(device), condition.to(device) 

            if config.use_decoder:

                # Forward pass through VAE
                reconstruction, mu, sigma, mask_u, mask_pu, norm_param_pred_u, norm_param_pred_pu = model.exact_reconstruction(X, u_proportion, pu_proportion, condition)
            else:
                mask_u, mask_pu, norm_param_pred_u, norm_param_pred_pu = model.exact_reconstruction(X, u_proportion, pu_proportion, condition)

            
            # Store results
            if config.use_decoder:
                mu_data.append(mu.cpu().detach().numpy())
                sigma_data.append(torch.exp(0.5 * sigma).cpu().detach().numpy())
                 # Calculate reconstruction error (MSE)
                reconstruction_error = ((X.cpu().detach().numpy() - reconstruction.cpu().detach().numpy()) ** 2).mean(axis=1)
                error_reconstruction.append(reconstruction_error)
                reconstruction_data.append(reconstruction.cpu().detach().numpy())

            truth_data.append(X.cpu().detach().numpy())
            
            if torch.sum(mask_u) > 0:
                norm_param_truth_u.append(u_proportion[mask_u].cpu().detach().numpy())
                norm_param_pred_u_list.append(norm_param_pred_u.cpu().detach().numpy())

            norm_param_truth_pu.append(pu_proportion[mask_pu].cpu().detach().numpy())
            norm_param_pred_pu_list.append(norm_param_pred_pu.cpu().detach().numpy())

            mask_u_list.append(mask_u.cpu().detach().numpy())
            mask_pu_list.append(mask_pu.cpu().detach().numpy())

           
    
    # Concatenate all batches
    results = {
        'truth': np.concatenate(truth_data, axis=0),
        'norm_param_truth_u': np.concatenate(norm_param_truth_u, axis=0),
        'norm_param_truth_pu': np.concatenate(norm_param_truth_pu, axis=0),
        'norm_param_pred_u': np.concatenate(norm_param_pred_u_list, axis=0),
        'norm_param_pred_pu': np.concatenate(norm_param_pred_pu_list, axis=0),
        'mask_u': np.concatenate(mask_u_list, axis=0),
        'mask_pu': np.concatenate(mask_pu_list, axis=0)
    }

    if config.use_decoder:
        results['mu'] = np.concatenate(mu_data, axis=0)
        results['sigma'] = np.concatenate(sigma_data, axis=0)
        results['error_reconstruction'] = np.concatenate(error_reconstruction, axis=0)
        results['reconstruction'] = np.concatenate(reconstruction_data, axis=0)

    return results


def create_visualizations(results_train, results_val, results_test, config, path):
    """Create various visualizations for model evaluation."""
    # Combine all results
    all_mu = np.concatenate([results_train['mu'], results_val['mu'], results_test['mu']], axis=0)
    all_sigma = np.concatenate([results_train['sigma'], results_val['sigma'], results_test['sigma']], axis=0)

    all_condition_u = np.concatenate([results_train['norm_param_truth_u'], results_val['norm_param_truth_u'], results_test['norm_param_truth_u']], axis=0)
    all_condition_pu = np.concatenate([results_train['norm_param_truth_pu'], results_val['norm_param_truth_pu'], results_test['norm_param_truth_pu']], axis=0)

    mask_u = np.concatenate([results_train['mask_u'], results_val['mask_u'], results_test['mask_u']], axis=0)
    mask_pu = np.concatenate([results_train['mask_pu'], results_val['mask_pu'], results_test['mask_pu']], axis=0)

    # Create t-SNE visualization of latent space
    print("Creating t-SNE visualization...")
    tsne = TSNE(n_components=2, random_state=42)
    latent_space_2d = tsne.fit_transform(all_mu)

    fig, ax = plt.subplots(1,2)
    mu_z = np.mean(all_mu, axis = 0)
    var_z = np.mean(all_sigma, axis = 0) + np.var(all_mu, axis = 0)
    print("***")
    print(mu_z.shape)
    print(var_z.shape)
    sns.histplot(mu_z, ax = ax[0])
    sns.histplot(var_z, ax = ax[1])
    ax[0].set_title("Mean Z space")
    ax[1].set_title("Sigma Z space")
    #fig.legend()
    fig.savefig(f"{path}/Z Distribution on set")
    plt.close()
    
    # Visualize latent space with different conditions
    
    # condition_names = ["234U", "235U", "236U", "238U", "238Pu", "239Pu", "240Pu", "241Pu", "242Pu"]#['Pu-238/Pu', 'Pu-239/Pu', 'Pu-240/Pu', 'Pu-241/Pu', 'Pu-242/Pu']

    condition_names_u = ["234U", "235U", "236U", "238U"]
    condition_types_u = ["quanti"]*len(condition_names_u)

    pu_condition_names_pu = ["238Pu", "239Pu", "240Pu", "241Pu", "242Pu"]
    condition_types_pu = ["quanti"]*len(pu_condition_names_pu)

    for n_condition, (condition_name, condition_type) in enumerate(zip(condition_names_u, condition_types_u)):
        new_condition_name = condition_name.replace("/", "-")

        labels_classe = np.unique(all_condition_u[:,n_condition])
        latent_space_visualisation(latent_space_2d[mask_u], all_condition_u[:,n_condition], labels_classe, f"{path}/LS visualisation hue {new_condition_name}", viridis = True, type = condition_type)
    
    for n_condition, (condition_name, condition_type) in enumerate(zip(pu_condition_names_pu, condition_types_pu)):
        
        new_condition_name = condition_name.replace("/", "-")

        labels_classe = np.unique(all_condition_pu[:,n_condition])
        latent_space_visualisation(latent_space_2d[mask_pu], all_condition_pu[:,n_condition], labels_classe, f"{path}/LS visualisation hue {new_condition_name}", viridis = True, type = condition_type)
    
    # Visualize by dataset split
    dataset_labels = np.array([0] * len(results_train['mu']) + 
                             [1] * len(results_val['mu']) + 
                             [2] * len(results_test['mu']))
    latent_space_visualisation(
        latent_space_2d, 
        dataset_labels, 
        ["train", "val", "test"], 
        f"{path}/LS_visualisation_train_val_test", 
        type="quali"
    )
    
    return latent_space_2d


def plot_r2_scores(results_train, results_val, results_test, config, path):
    """Plot R² scores for parameter prediction."""
    
    datasets = [
        (results_train, "train"),
        (results_val, "val"), 
        (results_test, "test")
    ]

    for results, dataset_name in datasets:        
        for i, isotop in  enumerate(["238Pu", "239Pu", "240Pu", "241Pu", "242Pu"]):
        

            isotop_name = isotop.replace("/", "-")
            # Single output case
            
            r2_score_val = r2_score(results['norm_param_truth_pu'][:,i], results['norm_param_pred_pu'][:,i])
            fig = plt.figure(figsize=(8, 5))

            plt.scatter(results['norm_param_truth_pu'][:,i], results['norm_param_pred_pu'][:,i])
            min_ = min(np.min(results['norm_param_truth_pu'][:,i]), np.min(results['norm_param_pred_pu'][:,i]))
            max_ = max(np.max(results['norm_param_truth_pu'][:,i]), np.max(results['norm_param_pred_pu'][:,i]))

            plt.plot([min_, max_], [min_, max_], color='r')
            plt.xlabel("Truth")
            plt.ylabel("Prediction")
            plt.title(f"R²: {r2_score_val:.4f}")
        
            plt.tight_layout()
            fig.savefig(f"{path}/R2_{isotop_name}_{dataset_name}.png", dpi=300, bbox_inches='tight')
            plt.close()

        for i, isotop in  enumerate(["234U", "235U", "236U", "238U"]):
        

            isotop_name = isotop.replace("/", "-")
            # Single output case
            
            r2_score_val = r2_score(results['norm_param_truth_u'][:,i], results['norm_param_pred_u'][:,i])
            fig = plt.figure(figsize=(8, 5))

            plt.scatter(results['norm_param_truth_u'][:,i], results['norm_param_pred_u'][:,i])
            min_ = min(np.min(results['norm_param_truth_u'][:,i]), np.min(results['norm_param_pred_u'][:,i]))
            max_ = max(np.max(results['norm_param_truth_u'][:,i]), np.max(results['norm_param_pred_u'][:,i]))

            plt.plot([min_, max_], [min_, max_], color='r')
            plt.xlabel("Truth")
            plt.ylabel("Prediction")
            plt.title(f"R²: {r2_score_val:.4f}")
        
            plt.tight_layout()
            fig.savefig(f"{path}/R2_{isotop_name}_{dataset_name}.png", dpi=300, bbox_inches='tight')
            plt.close()



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
        dataset_name = "IDB"
    else:
        dataset_name = f"{config_test.date}/{config_test.time}"

    # Load VAE configuration
    load_path_vae = f"../weights_IDB/{dataset_name}"
    config = load_config(load_path_vae)
    set_default_config_values(config)
  
    # Load datasets
    (train_dataset, val_dataset, test_dataset, 
     inputs_dim, inputs_class, nb_classes) = load_datasets(config)
    
    # Create data loaders
    train_loader = data.DataLoader(train_dataset, batch_size=config.batch_size, num_workers=config.num_workers)
    val_loader = data.DataLoader(val_dataset, batch_size=config.batch_size, num_workers=config.num_workers)
    test_loader = data.DataLoader(test_dataset, batch_size=config.batch_size, num_workers=config.num_workers)
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')
    
    # Load models
    model = load_models(config, load_path_vae, inputs_dim, inputs_class, device)
    
    # Print model summary
    print(f"Input class dimension: {inputs_class}")
    # print(summary(model, (inputs_dim,)))
    
    # Create output directory
    output_path = f"../outputs/IDB/{dataset_name}"
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
    results_val = evaluate_model(model, val_loader, config, device)
    results_test = evaluate_model(model, test_loader, config, device)
    
    # Create visualizations
    # detector_name = train_dataset.detector_name
    if config.use_decoder:

        latent_space_2d = create_visualizations(results_train, results_val, results_test, config, output_path)
        
    # Plot R² scores
    plot_r2_scores(results_train, results_val, results_test, config, output_path)
    
    # Plot some reconstruction examples

    if config.use_decoder:

        print("Creating reconstruction plots...")
        random_indices_train = np.random.choice(len(results_train['reconstruction']), size=5, replace=False)
        random_indices_val = np.random.choice(len(results_val['reconstruction']), size=5, replace=False)
        random_indices_test = np.random.choice(len(results_test['reconstruction']), size=5, replace=False)

        reconstruction_signal(
            results_train['reconstruction'][random_indices_train],
            results_train['truth'][random_indices_train],
            f"{output_path}/reconstruction_plots_train_normalized_signal"
        )

        reconstruction_signal(
            results_val['reconstruction'][random_indices_val],
            results_val['truth'][random_indices_val],
            f"{output_path}/reconstruction_plots_val_normalized_signal"
        )

        reconstruction_signal(
            results_test['reconstruction'][random_indices_test],
            results_test['truth'][random_indices_test],
            f"{output_path}/reconstruction_plots_test_normalized_signal"
        )
    
    r2_score_test_pu = []
    for i in range(5):
        r2_score_test_pu.append(r2_score(results_test['norm_param_truth_pu'][:,i], results_test['norm_param_pred_pu'][:,i]))
    
    r2_score_test_pu = np.mean(np.array(r2_score_test_pu))

    r2_score_test_u = []
    for i in range(4):
        r2_score_test_u.append(r2_score(results_test['norm_param_truth_u'][:,i], results_test['norm_param_pred_u'][:,i]))
    
    r2_score_test_u = np.mean(np.array(r2_score_test_u))

    # Write summary statistics
    with open(f"{output_path}/evaluation_summary.txt", "w") as f:
        f.write("VAE Model Evaluation Summary\n")
        f.write("=" * 30 + "\n\n")
        f.write(f"Latent dimension: {config.latent_dim}\n")

        if config.use_decoder:
            f.write("Reconstruction Errors (MSE on normalized data):\n")
            f.write(f"  Train: {np.mean(results_train['error_reconstruction']):.6f}\n")
            f.write(f"  Val:   {np.mean(results_val['error_reconstruction']):.6f}\n")
            f.write(f"  Test:  {np.mean(results_test['error_reconstruction']):.6f}\n\n")
        f.write(f" R2  Test Pu:  {r2_score_test_pu}\n\n")
        f.write(f" R2  Test U:  {r2_score_test_u}\n\n")

    print("=" * 50)
    print("Evaluation completed successfully!")
    print(f"Results saved to: {output_path}")
    print("=" * 50)


if __name__ == '__main__':
    main()