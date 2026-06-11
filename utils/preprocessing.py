import pandas as pd
import numpy as np
import torch
import json
from torch.utils import data


class Dataset_dataset(data.Dataset):
    """
    PyTorch Dataset class for handling spectral data with conditions.
    
    This dataset handles spectral content with associated conditions including type,
    enrichment, and attenuation parameters.
    """
    
    def __init__(self, json_file, config):
        """
        Initialize the dataset.
        
        Args:
            json_file (str): Path to JSON file containing the data
            config: Configuration object with dataset parameters
            train_max (bool): Whether to include normalization parameters in training
            before_norm (bool): Whether data is before normalization
        """
        # Load data from JSON file
        df = pd.read_json(json_file)

        # Store configuration parameters
        with open("../datasets/ESARDA/dataset Pu/normalization_metadata.json") as json_file:
            self.meta_data = json.load(json_file)

        self.min_amp_dataset = self.meta_data["global_min_log_amp"]
        self.max_amp_dataset = self.meta_data["global_max_log_amp"]
    
        # Set main data arrays
        self.data = df["content"]
        # self.type = df['type_quanti']
        
        # Define condition parameters

        if config.preprocessing_version == 2:
            self.condition_name = ['Detector quanti', 'FWHM at 208 keV (keV)','max_amplitude_log_norm']
        else:
            self.condition_name = ['Detector quanti', 'FWHM at 208 keV (keV)','max_amplitude_scaled']

        self.detector_dict = dict(zip(df['Detector quanti'], df['Detector']))
        self.detector_list = [self.detector_dict[i] for i in range(len(self.detector_dict))]

        self.condition = df[self.condition_name]

        # Define regressor outputs
        self.outputs_name = ['Pu-238/Pu', 'Pu-239/Pu', 'Pu-240/Pu', 'Pu-241/Pu', 'Pu-242/Pu']

        self.outputs = df[self.outputs_name]
        self.file_name = df["File name"].values[:,np.newaxis]

        # Create mapping between quantitative and qualitative detector names
        # self.detector_dict = dict(zip(df["type_quanti"], df["type"]))
        # self.detector_name = df["type"]
        
        # Store configuration and derived parameters
        self.config = config
        self.signal_length = len(self.data.iloc[0])
        self.inputs_class = 6  # Number of condition classes
        
    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (x, y) or (x, y, norm_param) depending on not_training setting
                x: Spectral data as FloatTensor
                y: Conditions [type, enrichment, attenuation] as FloatTensor
                norm_param: Normalization parameter (if not_training=True)
        """
        # Get spectral data and conditions
        x = self.data.iloc[idx]
        cond = self.condition.iloc[idx]
        y = self.outputs.iloc[idx]

        # Convert to PyTorch tensors
        x = torch.FloatTensor(x)
        cond = torch.tensor(cond.to_numpy(), dtype=torch.float32)
        y = torch.tensor(y.to_numpy(), dtype=torch.float32)

        return x, cond, y
    
    def undo_normalization(self, x, norm_param):
        """
        Reverse the normalization applied to the data.
        
        Args:
            x: Normalized data
            norm_param: Normalization parameter used
            
        Returns:
            Denormalized data in original scale
        """
        # Scale normalization parameter if using scaled max
        norm_param = norm_param * (self.max_amp_dataset - self.min_amp_dataset) + self.min_amp_dataset
        
        # Reverse normalization: multiply by norm_param then reverse log transformation
        denormalize_x = x * norm_param
        denormalize_x = np.exp(denormalize_x) - 1
        
        return denormalize_x