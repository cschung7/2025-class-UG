#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NVDA Feature Engineering Script
Creates comprehensive technical indicators using myTA.py functions
"""

import sys
import os
import pandas as pd
import numpy as np
from termcolor import colored
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import myTA
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from myTA import (
    get_ta_momentum,
    get_ta_trend,
    get_ta_volatility,
    get_ta_volume,
    remove_nan_zero
)

# Configuration - ALL CAPS variables
PERIODS = [5, 10, 13, 20, 50, 200]
TARGET_HORIZON = 1  # Days ahead to predict
DATA_DIR = "data"
INPUT_FILE = os.path.join(DATA_DIR, "nvda_raw.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "nvda_features.csv")
FEATURES_FILE = os.path.join(DATA_DIR, "nvda_features_only.csv")
TARGET_FILE = os.path.join(DATA_DIR, "nvda_target.csv")

def load_data():
    """Load raw NVDA data"""
    try:
        print(colored(f"[INFO] Loading data from {INPUT_FILE}...", "blue"))
        data = pd.read_csv(INPUT_FILE, index_col=0, parse_dates=True, encoding="utf-8")
        print(colored(f"[SUCCESS] Data loaded: {data.shape}", "green"))
        return data
    except Exception as e:
        print(colored(f"[ERROR] Failed to load data: {str(e)}", "red"))
        raise

def create_features(data):
    """Create technical indicators"""
    try:
        print(colored(f"[INFO] Creating technical indicators...", "blue"))
        print(colored(f"[INFO] Periods: {PERIODS}", "cyan"))
        
        # Make a copy to avoid modifying original
        df = data.copy()
        
        # Ensure column names are lowercase
        df.columns = [col.lower() for col in df.columns]
        
        # Create momentum indicators
        print(colored(f"[INFO] Generating momentum indicators...", "yellow"))
        df = get_ta_momentum(df, period=PERIODS)
        
        # Create trend indicators
        print(colored(f"[INFO] Generating trend indicators...", "yellow"))
        df = get_ta_trend(df, period=PERIODS)
        
        # Create volatility indicators
        print(colored(f"[INFO] Generating volatility indicators...", "yellow"))
        df = get_ta_volatility(df, period=PERIODS)
        
        # Create volume indicators
        print(colored(f"[INFO] Generating volume indicators...", "yellow"))
        df = get_ta_volume(df, period=PERIODS)
        
        # Handle NaN and infinite values
        print(colored(f"[INFO] Cleaning data (removing NaN and infinite values)...", "yellow"))
        df = remove_nan_zero(df)
        
        # Additional features: returns, volatility
        print(colored(f"[INFO] Creating additional features (returns, volatility)...", "yellow"))
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility_5'] = df['returns'].rolling(5).std()
        df['volatility_20'] = df['returns'].rolling(20).std()
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Drop rows with NaN (from rolling calculations)
        df = df.dropna()
        
        print(colored(f"[SUCCESS] Features created: {df.shape[1]} features", "green"))
        return df
        
    except Exception as e:
        print(colored(f"[ERROR] Failed to create features: {str(e)}", "red"))
        raise

def create_target(data):
    """Create target variable (future return)"""
    try:
        print(colored(f"[INFO] Creating target variable (future return)...", "blue"))
        
        # Create target: future return (TARGET_HORIZON days ahead)
        target = data['close'].shift(-TARGET_HORIZON) / data['close'] - 1
        target = target.dropna()
        
        # Also create direction target (1 for positive return, 0 for negative)
        target_direction = (target > 0).astype(int)
        
        print(colored(f"[SUCCESS] Target created: {len(target)} samples", "green"))
        print(colored(f"[INFO] Target statistics:", "cyan"))
        print(target.describe())
        
        return target, target_direction
        
    except Exception as e:
        print(colored(f"[ERROR] Failed to create target: {str(e)}", "red"))
        raise

def align_data_and_target(data, target, target_direction):
    """Align features and target, removing NaN"""
    try:
        print(colored(f"[INFO] Aligning features and target...", "blue"))
        
        # Align indices
        common_idx = data.index.intersection(target.index)
        data_aligned = data.loc[common_idx].copy()
        target_aligned = target.loc[common_idx].copy()
        target_dir_aligned = target_direction.loc[common_idx].copy()
        
        # Remove any remaining NaN
        mask = ~(data_aligned.isna().any(axis=1) | target_aligned.isna())
        data_final = data_aligned[mask].copy()
        target_final = target_aligned[mask].copy()
        target_dir_final = target_dir_aligned[mask].copy()
        
        print(colored(f"[SUCCESS] Final dataset: {data_final.shape[0]} samples, {data_final.shape[1]} features", "green"))
        
        return data_final, target_final, target_dir_final
        
    except Exception as e:
        print(colored(f"[ERROR] Failed to align data: {str(e)}", "red"))
        raise

def save_data(data, target, target_direction):
    """Save processed data"""
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # Save full dataset with target
        full_data = data.copy()
        full_data['target_return'] = target
        full_data['target_direction'] = target_direction
        full_data.to_csv(OUTPUT_FILE, encoding="utf-8")
        print(colored(f"[SUCCESS] Full dataset saved to: {OUTPUT_FILE}", "green"))
        
        # Save features only
        data.to_csv(FEATURES_FILE, encoding="utf-8")
        print(colored(f"[SUCCESS] Features saved to: {FEATURES_FILE}", "green"))
        
        # Save target
        target_df = pd.DataFrame({
            'target_return': target,
            'target_direction': target_direction
        }, index=target.index)
        target_df.to_csv(TARGET_FILE, encoding="utf-8")
        print(colored(f"[SUCCESS] Target saved to: {TARGET_FILE}", "green"))
        
    except Exception as e:
        print(colored(f"[ERROR] Failed to save data: {str(e)}", "red"))
        raise

def main():
    """Main function"""
    try:
        # Load data
        data = load_data()
        
        # Create features
        data_with_features = create_features(data)
        
        # Create target
        target, target_direction = create_target(data_with_features)
        
        # Align data and target
        features, target_aligned, target_dir_aligned = align_data_and_target(
            data_with_features, target, target_direction
        )
        
        # Display feature information
        print(colored(f"\n[INFO] Feature Summary:", "blue"))
        print(colored(f"  - Total features: {features.shape[1]}", "cyan"))
        print(colored(f"  - Total samples: {features.shape[0]}", "cyan"))
        print(colored(f"  - Feature columns: {list(features.columns[:10])}... (showing first 10)", "cyan"))
        
        # Save data
        save_data(features, target_aligned, target_dir_aligned)
        
        print(colored(f"\n[COMPLETE] Feature engineering finished!", "green", attrs=["bold"]))
        
        return features, target_aligned, target_dir_aligned
        
    except Exception as e:
        print(colored(f"[ERROR] Feature engineering failed: {str(e)}", "red"))
        raise

if __name__ == "__main__":
    features, target, target_dir = main()

