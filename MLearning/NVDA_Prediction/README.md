# NVDA Stock Prediction - ML vs DL Comparison

This project implements a comprehensive stock prediction system for NVDA (NVIDIA) using both traditional Machine Learning and Deep Learning approaches.

## Project Structure

```
NVDA_Prediction/
├── data/                          # Data storage
│   ├── nvda_raw.csv               # Raw downloaded data
│   ├── nvda_features.csv          # Features with target
│   ├── nvda_features_only.csv    # Features only
│   └── nvda_target.csv            # Target variables
├── models/
│   ├── ml_models/                 # Saved ML models
│   │   ├── randomforest.pkl
│   │   ├── xgboost.pkl
│   │   ├── svm.pkl
│   │   └── scaler.pkl
│   └── dl_models/                 # Saved DL models
│       ├── transformer.pth
│       ├── dilatedcnn.pth
│       └── lstm.pth
├── results/
│   ├── predictions/               # Model predictions
│   │   ├── ml_predictions.csv
│   │   └── dl_predictions.csv
│   ├── backtest_results/          # Backtesting results
│   │   ├── backtest_summary.csv
│   │   └── *_backtest_detail.csv
│   └── visualizations/            # Generated plots
│       ├── performance_metrics.png
│       ├── equity_curves.png
│       └── predictions_vs_actual.png
├── nvda_data_download.py          # Step 1: Download data
├── nvda_feature_engineering.py    # Step 2: Create features
├── nvda_ml_training.py            # Step 3: Train ML models
├── nvda_dl_training.py            # Step 4: Train DL models
├── nvda_backtesting.py            # Step 5: Backtest all models
└── nvda_comparison.py             # Step 6: Compare results
```

## Execution Order

Run the scripts in this order:

1. **Data Download**: `python3 nvda_data_download.py`
2. **Feature Engineering**: `python3 nvda_feature_engineering.py`
3. **ML Training**: `python3 nvda_ml_training.py`
4. **DL Training**: `python3 nvda_dl_training.py`
5. **Backtesting**: `python3 nvda_backtesting.py`
6. **Comparison**: `python3 nvda_comparison.py`

## Features

### Machine Learning Models
- **Random Forest**: Ensemble tree-based model
- **XGBoost**: Gradient boosting model
- **SVM**: Support Vector Machine (with RBF kernel)

### Deep Learning Models
- **Transformer**: Attention-based architecture for time series
- **Dilated CNN**: Convolutional network with dilated convolutions
- **LSTM**: Long Short-Term Memory network

### Technical Indicators
- Momentum: RSI, ROC, CMO, Williams %R
- Trend: SMA, EMA, WMA, TRIX, CCI, DPO, KST, ADX, DMI
- Volatility: Bollinger Bands, ATR
- Volume: CMF, MFI, Force Index, EOM

## Requirements

See `requirements.txt` for all dependencies.

## Notes

- All models use proper time-series splitting (no look-ahead bias)
- Real backtesting with actual Sharpe ratio calculations
- No mock data - all results from real predictions
- Transaction costs considered in backtesting

