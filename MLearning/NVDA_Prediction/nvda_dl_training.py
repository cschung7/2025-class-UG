#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NVDA Deep Learning Training Script
Trains three DL models: Transformer, CNN with Dilated Convolutions, LSTM
"""

import sys
import os
import pandas as pd
import numpy as np
from termcolor import colored
import time
import warnings
warnings.filterwarnings('ignore')

# Try to import PyTorch, fallback to TensorFlow if not available
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
        TF_AVAILABLE = True
    except ImportError:
        TF_AVAILABLE = False
        print(colored("[WARNING] Neither PyTorch nor TensorFlow available. Please install one.", "yellow"))

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configuration - ALL CAPS variables
DATA_DIR = "data"
MODELS_DIR = "models/dl_models"
FEATURES_FILE = os.path.join(DATA_DIR, "nvda_features_only.csv")
TARGET_FILE = os.path.join(DATA_DIR, "nvda_target.csv")

SEQUENCE_LENGTH = 60  # Days of history
BATCH_SIZE = 32
EPOCHS = 50  # Reduced for faster execution
LEARNING_RATE = 0.001
PATIENCE = 10  # Early stopping patience

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Device configuration
if TORCH_AVAILABLE:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(colored(f"[INFO] Using device: {DEVICE}", "cyan"))

class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series"""
    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class TransformerModel(nn.Module):
    """Transformer model for time series prediction"""
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1000, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        seq_len = x.size(1)
        x = self.input_projection(x)
        x = x + self.pos_encoder[:seq_len, :].unsqueeze(0)
        x = self.transformer(x)
        x = x[:, -1, :]  # Take last timestep
        x = self.dropout(x)
        x = self.fc(x)
        return x.squeeze(-1)

class DilatedCNNModel(nn.Module):
    """CNN with dilated convolutions for time series"""
    def __init__(self, input_dim, num_filters=64, dilation_rates=[1, 2, 4, 8], dropout=0.1):
        super(DilatedCNNModel, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(input_dim, num_filters, kernel_size=3, dilation=d, padding=d)
            for d in dilation_rates
        ])
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(num_filters) for _ in dilation_rates
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(num_filters * len(dilation_rates), 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x shape: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        
        conv_outputs = []
        for conv, bn in zip(self.convs, self.batch_norms):
            out = conv(x)
            out = bn(out)
            out = self.relu(out)
            out = torch.max(out, dim=2)[0]  # Global max pooling
            conv_outputs.append(out)
        
        x = torch.cat(conv_outputs, dim=1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze(-1)

class LSTMModel(nn.Module):
    """LSTM model for time series prediction"""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # Take last timestep
        x = self.dropout(x)
        x = self.fc(x)
        return x.squeeze(-1)

def create_sequences(data, target, seq_length):
    """Create sequences for time series prediction"""
    sequences = []
    targets = []
    
    for i in range(seq_length, len(data)):
        sequences.append(data[i-seq_length:i])
        targets.append(target[i])
    
    return np.array(sequences), np.array(targets)

def load_and_prepare_data():
    """Load and prepare data for deep learning"""
    try:
        print(colored(f"[INFO] Loading data...", "blue"))
        features = pd.read_csv(FEATURES_FILE, index_col=0, parse_dates=True, encoding="utf-8")
        target_df = pd.read_csv(TARGET_FILE, index_col=0, parse_dates=True, encoding="utf-8")
        target = target_df['target_return'].values
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Create sequences
        print(colored(f"[INFO] Creating sequences (length={SEQUENCE_LENGTH})...", "blue"))
        sequences, targets = create_sequences(features_scaled, target, SEQUENCE_LENGTH)
        
        print(colored(f"[SUCCESS] Created {len(sequences)} sequences", "green"))
        
        # Split data
        n = len(sequences)
        train_end = int(n * TRAIN_RATIO)
        val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
        
        X_train = sequences[:train_end]
        y_train = targets[:train_end]
        X_val = sequences[train_end:val_end]
        y_val = targets[train_end:val_end]
        X_test = sequences[val_end:]
        y_test = targets[val_end:]
        
        print(colored(f"[INFO] Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}", "cyan"))
        
        return X_train, X_val, X_test, y_train, y_val, y_test, scaler, features.shape[1]
        
    except Exception as e:
        print(colored(f"[ERROR] Failed to load data: {str(e)}", "red"))
        raise

def train_model(model, train_loader, val_loader, model_name, input_dim):
    """Train a deep learning model"""
    try:
        print(colored(f"\n[INFO] Training {model_name}...", "blue"))
        
        model = model.to(DEVICE)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        start_time = time.time()
        
        for epoch in range(EPOCHS):
            # Training
            model.train()
            train_loss = 0.0
            for sequences, targets in train_loader:
                sequences, targets = sequences.to(DEVICE), targets.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = model(sequences)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for sequences, targets in val_loader:
                    sequences, targets = sequences.to(DEVICE), targets.to(DEVICE)
                    outputs = model(sequences)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(colored(f"  Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}", "cyan"))
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(colored(f"  Early stopping at epoch {epoch+1}", "yellow"))
                    model.load_state_dict(best_model_state)
                    break
        
        train_time = time.time() - start_time
        print(colored(f"[SUCCESS] {model_name} trained in {train_time:.2f}s", "green"))
        
        return model, train_losses, val_losses, train_time
        
    except Exception as e:
        print(colored(f"[ERROR] Failed to train {model_name}: {str(e)}", "red"))
        raise

def evaluate_model(model, test_loader, model_name):
    """Evaluate model on test set"""
    try:
        model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for sequences, targets in test_loader:
                sequences = sequences.to(DEVICE)
                outputs = model(sequences)
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(targets.numpy())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        r2 = r2_score(actuals, predictions)
        
        print(colored(f"\n{model_name} Test Results:", "yellow"))
        print(colored(f"  MAE: {mae:.6f}, RMSE: {rmse:.6f}, R²: {r2:.4f}", "cyan"))
        
        return {
            'predictions': predictions,
            'actuals': actuals,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
        
    except Exception as e:
        print(colored(f"[ERROR] Failed to evaluate {model_name}: {str(e)}", "red"))
        raise

def main():
    """Main function"""
    if not TORCH_AVAILABLE:
        print(colored("[ERROR] PyTorch is required for deep learning models. Please install: pip install torch", "red"))
        return
    
    try:
        # Load and prepare data
        X_train, X_val, X_test, y_train, y_val, y_test, scaler, input_dim = load_and_prepare_data()
        
        # Create data loaders
        train_dataset = TimeSeriesDataset(X_train, y_train)
        val_dataset = TimeSeriesDataset(X_val, y_val)
        test_dataset = TimeSeriesDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        models = {}
        results = {}
        
        # Train Transformer
        transformer = TransformerModel(input_dim=input_dim)
        transformer, tr_losses, tr_val_losses, tr_time = train_model(
            transformer, train_loader, val_loader, "Transformer", input_dim
        )
        tr_results = evaluate_model(transformer, test_loader, "Transformer")
        models['Transformer'] = transformer
        results['Transformer'] = {
            **tr_results,
            'train_time': tr_time,
            'train_losses': tr_losses,
            'val_losses': tr_val_losses
        }
        
        # Train Dilated CNN
        cnn = DilatedCNNModel(input_dim=input_dim)
        cnn, cnn_losses, cnn_val_losses, cnn_time = train_model(
            cnn, train_loader, val_loader, "DilatedCNN", input_dim
        )
        cnn_results = evaluate_model(cnn, test_loader, "DilatedCNN")
        models['DilatedCNN'] = cnn
        results['DilatedCNN'] = {
            **cnn_results,
            'train_time': cnn_time,
            'train_losses': cnn_losses,
            'val_losses': cnn_val_losses
        }
        
        # Train LSTM
        lstm = LSTMModel(input_dim=input_dim)
        lstm, lstm_losses, lstm_val_losses, lstm_time = train_model(
            lstm, train_loader, val_loader, "LSTM", input_dim
        )
        lstm_results = evaluate_model(lstm, test_loader, "LSTM")
        models['LSTM'] = lstm
        results['LSTM'] = {
            **lstm_results,
            'train_time': lstm_time,
            'train_losses': lstm_losses,
            'val_losses': lstm_val_losses
        }
        
        # Save models
        os.makedirs(MODELS_DIR, exist_ok=True)
        for name, model in models.items():
            model_path = os.path.join(MODELS_DIR, f"{name.lower()}.pth")
            torch.save(model.state_dict(), model_path)
            print(colored(f"[SUCCESS] {name} saved to {model_path}", "green"))
        
        # Save predictions
        results_dir = "results/predictions"
        os.makedirs(results_dir, exist_ok=True)
        
        predictions_df = pd.DataFrame({
            'actual': results['Transformer']['actuals'],
            'transformer_pred': results['Transformer']['predictions'],
            'cnn_pred': results['DilatedCNN']['predictions'],
            'lstm_pred': results['LSTM']['predictions']
        })
        
        predictions_file = os.path.join(results_dir, "dl_predictions.csv")
        predictions_df.to_csv(predictions_file, encoding="utf-8")
        print(colored(f"\n[SUCCESS] Predictions saved to {predictions_file}", "green"))
        
        print(colored(f"\n[COMPLETE] Deep learning training finished!", "green", attrs=["bold"]))
        
        return models, results
        
    except Exception as e:
        print(colored(f"[ERROR] Deep learning training failed: {str(e)}", "red"))
        raise

if __name__ == "__main__":
    models, results = main()

