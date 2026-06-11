# -*- coding: utf-8 -*-
"""
XGBoost Regression Model with Hyperparameter Tuning
====================================================
This script trains an XGBoost regression model to predict max_scaled values
using detector type, enrichment, and counting time features.
"""

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib


def load_datasets():
    """
    Load training, validation, and test datasets from JSON files.
    
    Returns:
        tuple: (train_df, val_df, test_df) - pandas DataFrames
    """
    # Dataset paths
    base_path = '../datasets/ESARDA/dataset Pu'

    train_df = pd.read_json(f'{base_path}/train.json')
    val_df = pd.read_json(f'{base_path}/val.json')
    test_df = pd.read_json(f'{base_path}/test.json')
    
    return train_df, val_df, test_df


def prepare_features_and_targets(train_df, val_df, test_df):
    """
    Extract features and target variables from the datasets.
    
    Args:
        train_df, val_df, test_df: pandas DataFrames with the raw data
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Feature columns

    feature_cols = [
        'Detector quanti',           # Detector type (quantitative)
        'real counting times',
        'FWHM at 208 keV (keV)'
        ]
        
    
    # Target column
    target_col = 'max_amplitude_scaled'
    
    # Extract features
    X_train = train_df[feature_cols]
    X_val = val_df[feature_cols]
    X_test = test_df[feature_cols]
    
    # Extract targets
    y_train = train_df[target_col]
    y_val = val_df[target_col]
    y_test = test_df[target_col]
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def setup_xgboost_grid_search():
    """
    Set up XGBoost model and hyperparameter grid for grid search.
    
    Returns:
        tuple: (model, param_grid) - XGBoost model and parameter grid
    """
    # Initialize XGBoost regressor
    xgb_model = XGBRegressor(
        objective='reg:squarederror',  # Regression with squared error
        random_state=42               # For reproducible results
    )
    
    # Hyperparameter grid for tuning
    param_grid = {
        'n_estimators': [50, 100, 200],    # Number of boosting rounds
        'max_depth': [3, 5, 10],           # Maximum tree depth
        'learning_rate': [0.01, 0.1, 0.3], # Learning rate (eta)
        'subsample': [0.8, 1.0]            # Subsample ratio
    }
    
    return xgb_model, param_grid


def perform_grid_search(model, param_grid, X_train, y_train):
    """
    Perform grid search with cross-validation to find best hyperparameters.
    
    Args:
        model: XGBoost model
        param_grid: Dictionary of hyperparameters to test
        X_train, y_train: Training features and targets
        
    Returns:
        GridSearchCV: Fitted grid search object
    """
    # Set up grid search with 5-fold cross-validation
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,                    # 5-fold cross-validation
        scoring='r2',            # R-squared scoring metric
        n_jobs=-1,              # Use all available cores
        verbose=0               # Suppress output during training
    )
    
    # Fit the grid search
    print("Starting hyperparameter tuning...")
    grid_search.fit(X_train, y_train)
    print("Hyperparameter tuning completed!")
    
    return grid_search


def evaluate_model(grid_search, X_val, y_val, X_test, y_test):
    """
    Evaluate the best model on validation and test sets.
    
    Args:
        grid_search: Fitted GridSearchCV object
        X_val, y_val: Validation features and targets
        X_test, y_test: Test features and targets
    """
    # Make predictions
    y_pred_val = grid_search.predict(X_val)
    y_pred_test = grid_search.predict(X_test)
    
    # Calculate validation metrics
    rmse_val = mean_squared_error(y_val, y_pred_val, squared=False)
    r2_val = r2_score(y_val, y_pred_val)
    
    # Calculate test metrics
    rmse_test = mean_squared_error(y_test, y_pred_test, squared=False)
    r2_test = r2_score(y_test, y_pred_test)
    
    # Print results
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"Best hyperparameters: {grid_search.best_params_}")
    print(f"\nValidation Set Performance:")
    print(f"  RMSE: {rmse_val:.4f}")
    print(f"  R²:   {r2_val:.4f}")
    print(f"\nTest Set Performance:")
    print(f"  RMSE: {rmse_test:.4f}")
    print(f"  R²:   {r2_test:.4f}")
    print("="*50)


def save_model(grid_search, filename):
    """
    Save the best trained model to disk.
    
    Args:
        grid_search: Fitted GridSearchCV object
        filename: Name of the file to save the model
    """
    joblib.dump(grid_search.best_estimator_, filename)
    print(f"\nModel saved as: {filename}")


def main():
    """
    Main function to orchestrate the entire machine learning pipeline.
    """
    # Load datasets
    print("Loading datasets...")
    train_df, val_df, test_df = load_datasets()
    
    # Display dataset info
    print(f"Training set shape: {train_df.shape}")
    print(f"Validation set shape: {val_df.shape}")
    print(f"Test set shape: {test_df.shape}")
    print(f"Available columns: {list(train_df.columns)}")
    
    # Prepare features and targets
    print("\nPreparing features and targets...")
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_features_and_targets(
        train_df, val_df, test_df
    )
    
    # Set up model and parameter grid
    print("Setting up XGBoost model and parameter grid...")
    xgb_model, param_grid = setup_xgboost_grid_search()
    
    # Perform grid search
    grid_search = perform_grid_search(xgb_model, param_grid, X_train, y_train)
    
    # Evaluate the best model
    evaluate_model(grid_search, X_val, y_val, X_test, y_test)
    
    # Save the trained model
    filename="../weights/regression_max.joblib"

    save_model(grid_search, filename=filename)


if __name__ == "__main__":
    main()


# Alternative configurations (commented out for reference)
# ========================================================

# Dataset 1 configuration (uncomment to use):
# train_df = pd.read_json('../MC/train 3.json')
# val_df = pd.read_json('../MC/val 3.json')
# test_df = pd.read_json('../MC/test 3.json')
# feature_cols = ['type_quanti', 'enrichment', 'attenuation']

# Random Forest alternative (uncomment to use instead of XGBoost):
# rf_model = RandomForestRegressor(random_state=42)
# rf_param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [None, 5, 10, 20],
#     'min_samples_split': [2, 5, 10]
# }