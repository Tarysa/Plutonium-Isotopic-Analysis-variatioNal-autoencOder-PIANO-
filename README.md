# A novel framework for qualitative and quantitative isotopic analysis of plutonium materials using VAE
### A feasibility study using ESARDA and IAEA-IDB databases

**Authors**: T. Sendra, I. Meleshenkovskii, E. Mauerhofer, V. Vigneron

Official implementation of the isotopic analysis code **Plutonium Isotopic Analysis variatioNal autoencOder (PIANO)**.

## Abstract

**Background:** The determination of plutonium isotopic composition from γ-ray energy spectra is essential for nuclear safeguards and material characterization. However, spectral features depend not only on isotopic composition but also on detector response and measurement conditions, which complicates the analysis, particularly for medium-resolution detectors.

**Purpose:** This work proposes a data-driven framework to extract isotopic composition information from γ-spectra of plutonium bearing materials while accounting for detector characteristics and measurement conditions.

**Methods:** The proposed method is based on a VAE architecture implemented within the isotopic analysis code PIANO. The model incorporates explicit parametrization of detector types and measurement conditions, enabling it to capture the relationships between spectrometric features and isotopic composition. Training and validation were conducted using plutonium-X and γ-ray spectra from the ESARDA and the IDB databases acquired with high-resolution HPGe detectors and medium-resolution room-temperature detectors such as CZT and others.

## 📁 Project Structure

```text
code_v2/
├── datasets/              # Input datasets
│   ├── ESARDA/
│   └── IDB/
├── models/                # Model architectures and related modules
│   ├── GRL.py  
│   ├── model.py           # ESARDA models
│   └── model_IDB*.py      # IDB models
├── outputs/               # Generated outputs (e.g., predictions, plots)
├── runs/                  # TensorBoard logs and run metadata
├── train_eval/            # Training and evaluation scripts
│   ├── regressor.py  
│   ├── script*.sh         # Batch scripts for training/testing
│   ├── test*.py           # Evaluation scripts
│   └── train*.py          # Training scripts
├── utils/                 # Utility functions and preprocessing
│   ├── functions.py  
│   └── preprocessing*.py  # Dataset-specific preprocessing
├── weights/               # Pretrained model weights (ESARDA)
└── weights_IDB/           # Pretrained model weights (IDB)
```

## 🚀 Getting Started

### 1. Install dependencies

Recommended: Create and activate a virtual environment

You can install dependencies with:

```bash
pip install -r requirements.txt
```

### 2. Training

To train a model, use one of the following scripts:

For ESARDA dataset:
```bash
python3 train.py --dataset 1 --beta 2 --lr 1e-4 --n_epochs 20000 --latent_dim 10 --opt "adamw" --batch_size 4 --resnet True --nb_conv_layer 3 --reduction "sum" --loss "mse" --version 3 --beta_schedule_ratio 0.4 --weight_decay 5e-2 --film_layer True --beta_regressor 1 --type "batchnorm" --nb_dense_layer 2 --model_version 2 --alpha 0.6 
```

For IDB dataset (Pu only):
```bash
python3 train.py --dataset 3 --beta 0.1 --lr 1e-4 --n_epochs 20000 --latent_dim 10 --opt "adamw" --batch_size 6 --resnet True --nb_conv_layer 3 --reduction "sum" --loss "mse" --version 3 --beta_schedule_ratio 0.4 --weight_decay 5e-2 --film_layer True --beta_regressor 0.5 --type "batchnorm" --nb_dense_layer 2 --reduced_dataset True --model_version 2 --alpha 1 
```

### 3. Testing

Evaluate the model with:

For the ESARDA database:
```bash
python3 test.py --date 260217 --time "014612"
```

For the IDB database (Pu only):
```bash
python3 test_IDB_only_Pu.py --date 260228 --time "002212"
```

Make sure the corresponding model weights are available in the `weights/` directory.

## License

*(License information to be added)*

## Authors

T. Sendra, I. Meleshenkovskii, E. Mauerhofer, V. Vigneron
