#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NVDA Machine Learning Training Script
Trains multiple ML models: Random Forest, XGBoost, SVM
"""

import sys
import os
import pandas as pd
import numpy as np
from termcolor import colored
import pickle
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Configuration - ALL CAPS variables
DATA_DIR = "data"
MODELS_DIR = "models/ml_models"
FEATURES_FILE = os.path.join(DATA_DIR, "nvda_features_only.csv")
TARGET_FILE = os.path.join(DATA_DIR, "nvda_target.csv")

# Train/Val/Test split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Model parameters
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'random_state': 42
}

XGB_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'random_state': 42
}

SVM_PARAMS = {
    'kernel': 'rbf',
    'C': 1.0,
    'epsilon': 0.1
}

def load_data():
    """Load features and target"""
    try:
        print(colored(f"[INFO] Loading features and target...", "blue"))
        features = pd.read_csv(FEATURES_FILE, index_col=0, parse_dates=True, encoding="utf-8")
        target_df = pd.read_csv(TARGET_FILE, index_col=0, parse_dates=True, encoding="utf-8")
        target = target_df['target_return'].values
        
        print(colored(f"[SUCCESS] Data loaded: {features.shape[0]} samples, {features.shape[1]} features", "green"))
        return features, target
        
    except Exception as e:
        print(colored(f"[ERROR] Failed to load data: {str(e)}", "red"))
        raise

def time_series_split(X, y):
    """Split data into train/val/test with proper time-series ordering"""
    try:
        n = len(X)
        train_end = int(n * TRAIN_RATIO)
        val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
        
        X_train = X.iloc[:train_end]
        y_train = y[:train_end]
        X_val = X.iloc[train_end:val_end]
        y_val = y[train_end:val_end]
        X_test = X.iloc[val_end:]
        y_test = y[val_end:]
        
        print(colored(f"[INFO] Data split:", "blue"))
        print(colored(f"  Train: {len(X_train)} samples ({len(X_train)/n*100:.1f}%)", "cyan"))
        print(colored(f"  Val: {len(X_val)} samples ({len(X_val)/n*100:.1f}%)", "cyan"))
        print(colored(f"  Test: {len(X_test)} samples ({len(X_test)/n*100:.1f}%)", "cyan"))
        
        return X_train, X_val, X_test, y_train, y_val, y_test
        
    except Exception as e:
        print(colored(f"[ERROR] Failed to split data: {str(e)}", "red"))
        raise

def scale_features(X_train, X_val, X_test):
    """Scale features using StandardScaler"""
    try:
        print(colored(f"[INFO] Scaling features...", "blue"))
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            index=X_train.index,
            columns=X_train.columns
        )
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val),
            index=X_val.index,
            columns=X_val.columns
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            index=X_test.index,
            columns=X_test.columns
        )
        print(colored(f"[SUCCESS] Features scaled", "green"))
        return X_train_scaled, X_val_scaled, X_test_scaled, scaler
        
    except Exception as e:
        print(colored(f"[ERROR] Failed to scale features: {str(e)}", "red"))
        raise

def train_random_forest(X_train, y_train, X_val, y_val):
    """Train Random Forest model"""
    try:
        print(colored(f"\n[INFO] Training Random Forest...", "blue"))
        start_time = time.time()
        
        model = RandomForestRegressor(**RF_PARAMS, n_jobs=-1)
        model.fit(X_train, y_train)
        
        train_time = time.time() - start_time
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        # Metrics
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_r2 = r2_score(y_val, y_val_pred)
        
        print(colored(f"[SUCCESS] Random Forest trained in {train_time:.2f}s", "green"))
        print(colored(f"  Train - MAE: {train_mae:.6f}, RMSE: {train_rmse:.6f}, R²: {train_r2:.4f}", "cyan"))
        print(colored(f"  Val   - MAE: {val_mae:.6f}, RMSE: {val_rmse:.6f}, R²: {val_r2:.4f}", "cyan"))
        
        return model, {
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'train_r2': train_r2,
            'val_mae': val_mae,
            'val_rmse': val_rmse,
            'val_r2': val_r2,
            'train_time': train_time
        }
        
    except Exception as e:
        print(colored(f"[ERROR] Failed to train Random Forest: {str(e)}", "red"))
        raise

def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost model"""
    try:
        print(colored(f"\n[INFO] Training XGBoost...", "blue"))
        start_time = time.time()
        
        model = xgb.XGBRegressor(**XGB_PARAMS, n_jobs=-1)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        train_time = time.time() - start_time
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        # Metrics
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_r2 = r2_score(y_val, y_val_pred)
        
        print(colored(f"[SUCCESS] XGBoost trained in {train_time:.2f}s", "green"))
        print(colored(f"  Train - MAE: {train_mae:.6f}, RMSE: {train_rmse:.6f}, R²: {train_r2:.4f}", "cyan"))
        print(colored(f"  Val   - MAE: {val_mae:.6f}, RMSE: {val_rmse:.6f}, R²: {val_r2:.4f}", "cyan"))
        
        return model, {
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'train_r2': train_r2,
            'val_mae': val_mae,
            'val_rmse': val_rmse,
            'val_r2': val_r2,
            'train_time': train_time
        }
        
    except Exception as e:
        print(colored(f"[ERROR] Failed to train XGBoost: {str(e)}", "red"))
        raise

def train_svm(X_train, y_train, X_val, y_val):
    """Train SVM model"""
    try:
        print(colored(f"\n[INFO] Training SVM (this may take a while)...", "blue"))
        start_time = time.time()
        
        # Use subset for faster training if dataset is large
        if len(X_train) > 10000:
            sample_size = 10000
            indices = np.random.choice(len(X_train), sample_size, replace=False)
            X_train_sample = X_train.iloc[indices]
            y_train_sample = y_train[indices]
            print(colored(f"[INFO] Using {sample_size} samples for SVM training", "yellow"))
        else:
            X_train_sample = X_train
            y_train_sample = y_train
        
        model = SVR(**SVM_PARAMS)
        model.fit(X_train_sample, y_train_sample)
        
        train_time = time.time() - start_time
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        
        # Metrics
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_r2 = r2_score(y_val, y_val_pred)
        
        print(colored(f"[SUCCESS] SVM trained in {train_time:.2f}s", "green"))
        print(colored(f"  Train - MAE: {train_mae:.6f}, RMSE: {train_rmse:.6f}, R²: {train_r2:.4f}", "cyan"))
        print(colored(f"  Val   - MAE: {val_mae:.6f}, RMSE: {val_rmse:.6f}, R²: {val_r2:.4f}", "cyan"))
        
        return model, {
            'train_mae': train_mae,
            'train_rmse': train_rmse,
            'train_r2': train_r2,
            'val_mae': val_mae,
            'val_rmse': val_rmse,
            'val_r2': val_r2,
            'train_time': train_time
        }
        
    except Exception as e:
        print(colored(f"[ERROR] Failed to train SVM: {str(e)}", "red"))
        raise

def evaluate_test_set(models, X_test, y_test):
    """Evaluate all models on test set"""
    try:
        print(colored(f"\n[INFO] Evaluating models on test set...", "blue"))
        results = {}
        
        for name, model in models.items():
            y_test_pred = model.predict(X_test)
            
            test_mae = mean_absolute_error(y_test, y_test_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            test_r2 = r2_score(y_test, y_test_pred)
            
            results[name] = {
                'test_mae': test_mae,
                'test_rmse': test_rmse,
                'test_r2': test_r2,
                'predictions': y_test_pred
            }
            
            print(colored(f"\n{name} Test Results:", "yellow"))
            print(colored(f"  MAE: {test_mae:.6f}, RMSE: {test_rmse:.6f}, R²: {test_r2:.4f}", "cyan"))
        
        return results
        
    except Exception as e:
        print(colored(f"[ERROR] Failed to evaluate test set: {str(e)}", "red"))
        raise

def save_models(models, scaler):
    """Save trained models"""
    try:
        os.makedirs(MODELS_DIR, exist_ok=True)
        
        for name, model in models.items():
            model_file = os.path.join(MODELS_DIR, f"{name.lower().replace(' ', '_')}.pkl")
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            print(colored(f"[SUCCESS] {name} saved to {model_file}", "green"))
        
        scaler_file = os.path.join(MODELS_DIR, "scaler.pkl")
        with open(scaler_file, 'wb') as f:
            pickle.dump(scaler, f)
        print(colored(f"[SUCCESS] Scaler saved to {scaler_file}", "green"))
        
    except Exception as e:
        print(colored(f"[ERROR] Failed to save models: {str(e)}", "red"))
        raise

def main():
    """Main function"""
    try:
        # Load data
        features, target = load_data()
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = time_series_split(features, target)
        
        # Scale features
        X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(
            X_train, X_val, X_test
        )
        
        # Train models
        models = {}
        metrics = {}
        
        # Random Forest
        rf_model, rf_metrics = train_random_forest(X_train, y_train, X_val, y_val)
        models['RandomForest'] = rf_model
        metrics['RandomForest'] = rf_metrics
        
        # XGBoost
        xgb_model, xgb_metrics = train_xgboost(X_train, y_train, X_val, y_val)
        models['XGBoost'] = xgb_model
        metrics['XGBoost'] = xgb_metrics
        
        # SVM (use scaled features)
        svm_model, svm_metrics = train_svm(X_train_scaled, y_train, X_val_scaled, y_val)
        models['SVM'] = svm_model
        metrics['SVM'] = svm_metrics
        
        # Evaluate on test set
        test_results = evaluate_test_set(models, X_test, y_test)
        
        # Save models
        save_models(models, scaler)
        
        # Save predictions and results
        results_dir = "results/predictions"
        os.makedirs(results_dir, exist_ok=True)
        
        predictions_df = pd.DataFrame({
            'actual': y_test,
            'rf_pred': test_results['RandomForest']['predictions'],
            'xgb_pred': test_results['XGBoost']['predictions'],
            'svm_pred': test_results['SVM']['predictions']
        }, index=X_test.index)
        
        predictions_file = os.path.join(results_dir, "ml_predictions.csv")
        predictions_df.to_csv(predictions_file, encoding="utf-8")
        print(colored(f"\n[SUCCESS] Predictions saved to {predictions_file}", "green"))
        
        print(colored(f"\n[COMPLETE] ML training finished!", "green", attrs=["bold"]))
        
        return models, metrics, test_results
        
    except Exception as e:
        print(colored(f"[ERROR] ML training failed: {str(e)}", "red"))
        raise

if __name__ == "__main__":
    models, metrics, test_results = main()

