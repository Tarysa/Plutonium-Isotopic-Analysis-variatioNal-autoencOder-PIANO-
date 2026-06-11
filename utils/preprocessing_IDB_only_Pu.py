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
    
    def __init__(self, json_file, config, mode = "train"):
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

       
        print(df[["Detector quanti", "Detector"]])
        # df = df[df['Detector quanti'] < 4]

        # Store configuration parameters
        with open("../datasets/ESARDA/dataset Pu/normalization_metadata.json") as json_file:
            self.meta_data = json.load(json_file)
        
        # Set main data arrays
        self.data = df["content"]
        self.data.reset_index(drop=True, inplace=True)

        self.mode = mode
        # self.type = df['type_quanti']
        
       
        # Define condition parameters
        self.condition_name = ["Detector quanti", "max_amplitude_scaled", "ENER_FIT"]
        self.condition_name2 = ["Detector quanti", "max_amplitude", "ENER_FIT"]

        self.detector_dict = {k: v for k, v in zip(df["Detector quanti"], df["Detector"])}

        if config.only_Pu:
            self.iso_condition_name = ["238Pu", "239Pu", "240Pu", "241Pu", "242Pu"]
        elif config.only_U:
            self.iso_condition_name = ["234U", "235U", "236U", "238U"]

        print(df["Detector quanti"].value_counts())

        self.condition = df[self.condition_name].values
        self.condition2 = df[self.condition_name2].values
        self.iso_condition = df[self.iso_condition_name].values

        self.file_name = df['File name'].values[:,np.newaxis]

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
        
        condition = self.condition[idx]
        pu = self.iso_condition[idx]

        # Convert to PyTorch tensors
        x = torch.FloatTensor(x)

        condition = torch.FloatTensor(condition.copy())
        pu = torch.FloatTensor(pu.copy())

        if self.mode == "test":
            condition2 = self.condition2[idx]
            condition2 = torch.FloatTensor(condition2.copy())
            return x, pu, condition, condition2
        else:
            return x, pu, condition
    
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