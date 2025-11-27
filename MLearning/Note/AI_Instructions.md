# AI Instructions: NVDA Stock Prediction with ML and Deep Learning

## Overview
This document provides step-by-step instructions for building a comprehensive stock prediction system for NVDA (NVIDIA) using both traditional Machine Learning and Deep Learning approaches, followed by performance comparison.

---

## Phase 1: Data Preparation and Machine Learning

### Step 1: Download NVDA Data

**Objective**: Download historical NVDA stock data using yfinance.

**Requirements**:
- Use `yfinance` library to download NVDA data
- Download data from a reasonable start date (e.g., '2010-1-1' or '2015-1-1')
- Include OHLCV (Open, High, Low, Close, Volume) data
- Save raw data for reference
- Use termcolor for status updates
- Implement proper error handling with try-except blocks

**Implementation Guidelines**:
```python
import yfinance as yf
from termcolor import colored
import pandas as pd

TICKER = "NVDA"
START_DATE = "2010-01-01"
END_DATE = None  # Current date

# Download with status updates
# Handle errors gracefully
# Display data shape and basic statistics
```

**Expected Output**:
- Raw NVDA data saved (optional: CSV backup)
- Data shape and date range displayed
- Basic statistics printed

---

### Step 2: Feature Engineering

**Objective**: Create comprehensive technical indicators and features for prediction.

**Requirements**:
- Use the existing `myTA.py` functions for technical analysis
- Generate features from multiple categories:
  - **Momentum indicators**: RSI, ROC, CMO, Williams %R
  - **Trend indicators**: SMA, EMA, WMA, TRIX, CCI, DPO, KST, ADX, DMI
  - **Volatility indicators**: Bollinger Bands, ATR
  - **Volume indicators**: CMF, MFI, Force Index, EOM
- Create multiple period variations (e.g., [5, 10, 13, 20, 50, 200])
- Handle missing values and infinite values appropriately
- Create target variable (e.g., next day return, direction, or volatility)
- Remove or handle non-stationary features if needed

**Implementation Guidelines**:
```python
from myTA import (
    get_ta_momentum, 
    get_ta_trend, 
    get_ta_volatility, 
    get_ta_volume,
    remove_nan_zero
)

PERIODS = [5, 10, 13, 20, 50, 200]  # Define at top in ALL CAPS
TARGET_HORIZON = 1  # Days ahead to predict

# Apply all TA functions
# Create target variable (e.g., future return or direction)
# Handle NaN and infinite values
# Split into features (X) and target (y)
```

**Expected Output**:
- Feature matrix with all technical indicators
- Target variable created
- Clean dataset ready for ML models
- Feature count and statistics displayed

---

### Step 3: Machine Learning Model Selection and Prediction

**Objective**: Train and evaluate multiple ML models for stock prediction.

**Requirements**:
- Implement train/validation/test split (e.g., 70/15/15 or 60/20/20)
- **NO LOOK-AHEAD BIAS**: Ensure proper time-series split
- Test multiple ML models:
  - Random Forest
  - Gradient Boosting (XGBoost or LightGBM)
  - Support Vector Machine (SVM)
  - Logistic Regression (for classification tasks)
  - Ensemble methods (Stacking, Voting)
- Perform hyperparameter tuning (GridSearchCV or RandomizedSearchCV)
- Evaluate models using appropriate metrics:
  - For regression: MAE, RMSE, R², Sharpe Ratio (if applicable)
  - For classification: Accuracy, Precision, Recall, F1-Score
- Calculate actual backtesting performance:
  - Generate trading signals from predictions
  - Calculate returns, Sharpe ratio, max drawdown
  - **NO MOCK DATA**: All metrics must come from real predictions

**Implementation Guidelines**:
```python
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# Time-series split (no random shuffle)
# Hyperparameter tuning
# Model training with termcolor status updates
# Prediction generation
# Backtesting with real signals
# Performance metrics calculation
```

**Expected Output**:
- Trained ML models saved (optional: pickle files)
- Model performance metrics for each algorithm
- Backtesting results (returns, Sharpe ratio, max drawdown)
- Feature importance analysis
- Prediction results on test set

---

## Phase 2: Deep Learning Implementation

### Step 4: Deep Learning Models

**Objective**: Implement and train deep learning models for stock prediction.

**Requirements**:
- Prepare data for deep learning (sequence creation, normalization)
- Implement three deep learning architectures:

#### 4.1 Transformer Model
- Use attention mechanism for time-series prediction
- Multi-head self-attention layers
- Positional encoding for temporal information
- Encoder-decoder or encoder-only architecture
- Proper sequence length handling

#### 4.2 CNN with Dilated Convolutions
- Dilated convolutions for capturing long-term dependencies
- Multiple dilation rates (e.g., 1, 2, 4, 8)
- 1D convolutions for time-series data
- Residual connections if applicable
- Global pooling or flattening before final layers

#### 4.3 LSTM Model
- Long Short-Term Memory networks
- Stacked LSTM layers (2-3 layers)
- Bidirectional LSTM option
- Dropout for regularization
- Dense layers for final prediction

**Implementation Guidelines**:
```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Or use TensorFlow/Keras:
# from tensorflow import keras
# from tensorflow.keras import layers

SEQUENCE_LENGTH = 60  # Days of history
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001

# Data normalization (StandardScaler or MinMaxScaler)
# Sequence creation (sliding window)
# Model architecture definition
# Training loop with early stopping
# Model evaluation
```

**Model Architecture Requirements**:
- Input: Sequences of features (shape: [batch, sequence_length, num_features])
- Output: Prediction (regression: single value, classification: probabilities)
- Loss function: MSE for regression, CrossEntropy for classification
- Optimizer: Adam or AdamW
- Learning rate scheduling (optional but recommended)
- Early stopping based on validation loss

**Expected Output**:
- Trained deep learning models (saved checkpoints)
- Training history (loss curves)
- Validation and test set predictions
- Model architecture summaries
- Training time and resource usage

---

### Step 5: Backtesting Deep Learning Models

**Objective**: Evaluate deep learning models using real backtesting.

**Requirements**:
- Generate trading signals from DL model predictions
- Implement proper backtesting framework:
  - Walk-forward validation or expanding window
  - Transaction costs consideration (optional)
  - Position sizing (optional)
- Calculate performance metrics:
  - Total returns
  - Annualized returns
  - Sharpe ratio
  - Maximum drawdown
  - Win rate (if applicable)
- **NO MOCK DATA**: All metrics from actual predictions and signals

**Implementation Guidelines**:
```python
# Convert predictions to trading signals
# Calculate returns from signals
# Compute Sharpe ratio: mean(returns) / std(returns) * sqrt(252)
# Calculate max drawdown
# Compare with buy-and-hold strategy
```

**Expected Output**:
- Backtesting results for each DL model
- Performance comparison table
- Equity curves visualization
- Risk metrics (Sharpe, max drawdown)

---

## Phase 3: Comparison and Analysis

### Step 6: Compare Machine Learning vs Deep Learning Results

**Objective**: Comprehensive comparison of ML and DL model performance.

**Requirements**:
- Create comparison table/metrics:
  - Model name
  - Prediction accuracy (MAE, RMSE, or classification metrics)
  - Backtesting returns
  - Sharpe ratio
  - Maximum drawdown
  - Training time
  - Inference time
- Visualizations:
  - Performance comparison charts (bar plots, line plots)
  - Equity curves for all models
  - Prediction vs actual plots
  - Feature importance (for ML models)
  - Attention weights visualization (for Transformer, if applicable)
- Statistical significance testing (optional):
  - Compare model returns using t-tests
  - Diebold-Mariano test for forecast accuracy

**Implementation Guidelines**:
```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Create results DataFrame
# Generate comparison visualizations
# Statistical tests if applicable
# Summary report generation
```

**Expected Output**:
- Comprehensive comparison table
- Visualization plots (saved as images)
- Summary report with key findings
- Recommendations on best-performing model(s)

---

## Technical Requirements

### Code Standards
1. **Termcolor Printing**: Use termcolor for all status updates and progress tracking
2. **File Encoding**: Always use `encoding="utf-8"` for file operations
3. **Variables**: Define major variables in ALL CAPS at the top of scripts
4. **Error Handling**: Implement try-except blocks with descriptive error messages
5. **Separation of Concerns**: Keep data loading, feature engineering, model training, and evaluation in separate functions/classes

### Data Requirements
- **NO MOCK DATA**: All results must come from real data and actual model predictions
- **NO LOOK-AHEAD BIAS**: Proper time-series splitting
- **Real Backtesting**: Calculate actual Sharpe ratios from trading signals
- **Proper Data Splits**: Train/Val/Test with chronological order

### Model Requirements
- **Real Machine Learning**: All model weights must be learned from real market data
- **Proper Training**: Use validation sets for hyperparameter tuning
- **Risk Management**: Report max drawdown and consider position limits
- **No Artificial Performance**: All metrics from actual predictions

### Libraries and Dependencies
- Data: `pandas`, `numpy`, `yfinance`
- ML: `scikit-learn`, `xgboost`, `lightgbm`
- DL: `pytorch` or `tensorflow/keras`
- Visualization: `matplotlib`, `seaborn`, `plotly` (optional)
- Technical Analysis: Use existing `myTA.py` functions
- Utilities: `termcolor` for status updates

---

## File Structure

```
MLearning/
├── Note/
│   └── AI_Instructions.md (this file)
├── NVDA_Prediction/
│   ├── data/
│   │   └── nvda_raw.csv (optional backup)
│   ├── models/
│   │   ├── ml_models/ (saved ML models)
│   │   └── dl_models/ (saved DL models)
│   ├── results/
│   │   ├── predictions/
│   │   ├── backtest_results/
│   │   └── visualizations/
│   ├── nvda_data_download.py
│   ├── nvda_feature_engineering.py
│   ├── nvda_ml_training.py
│   ├── nvda_dl_training.py
│   ├── nvda_backtesting.py
│   └── nvda_comparison.py
└── myTA.py (existing)
```

---

## Execution Order

1. **Data Download**: Run `nvda_data_download.py`
2. **Feature Engineering**: Run `nvda_feature_engineering.py`
3. **ML Training**: Run `nvda_ml_training.py`
4. **DL Training**: Run `nvda_dl_training.py` (includes all three architectures)
5. **Backtesting**: Run `nvda_backtesting.py` (for both ML and DL)
6. **Comparison**: Run `nvda_comparison.py` (generates final report)

---

## Success Criteria

- ✅ NVDA data successfully downloaded and validated
- ✅ Comprehensive feature set created (100+ features from TA indicators)
- ✅ Multiple ML models trained and evaluated
- ✅ Three DL architectures (Transformer, CNN-dilation, LSTM) implemented
- ✅ Real backtesting performed for all models
- ✅ Performance metrics calculated (Sharpe, max drawdown, returns)
- ✅ Comprehensive comparison report generated
- ✅ All code follows project standards (termcolor, error handling, etc.)
- ✅ No mock data or artificial performance metrics

---

## Notes

- Adjust hyperparameters based on validation performance
- Consider feature selection if feature count is too high
- Monitor for overfitting (large gap between train and validation performance)
- Consider ensemble of best-performing models
- Document any assumptions or limitations in the final report

