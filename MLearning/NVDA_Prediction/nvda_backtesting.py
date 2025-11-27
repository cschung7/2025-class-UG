#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NVDA Backtesting Script
Performs backtesting on ML and DL model predictions
"""

import sys
import os
import pandas as pd
import numpy as np
from termcolor import colored
import warnings
warnings.filterwarnings('ignore')

# Configuration - ALL CAPS variables
DATA_DIR = "data"
RESULTS_DIR = "results/backtest_results"
PREDICTIONS_DIR = "results/predictions"

ML_PREDICTIONS_FILE = os.path.join(PREDICTIONS_DIR, "ml_predictions.csv")
DL_PREDICTIONS_FILE = os.path.join(PREDICTIONS_DIR, "dl_predictions.csv")
FEATURES_FILE = os.path.join(DATA_DIR, "nvda_features_only.csv")

TRANSACTION_COST = 0.001  # 0.1% transaction cost (optional)

def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """Calculate annualized Sharpe ratio"""
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    excess_returns = returns - risk_free_rate / 252
    sharpe = np.sqrt(252) * np.mean(excess_returns) / np.std(returns)
    return sharpe

def calculate_max_drawdown(equity_curve):
    """Calculate maximum drawdown"""
    peak = equity_curve.expanding().max()
    drawdown = (equity_curve - peak) / peak
    return drawdown.min()

def generate_signals(predictions, threshold=0.0):
    """Generate trading signals from predictions"""
    # Simple strategy: buy if predicted return > threshold, sell if < -threshold
    signals = np.where(predictions > threshold, 1, 
                      np.where(predictions < -threshold, -1, 0))
    return signals

def backtest_strategy(predictions, actual_returns, model_name, threshold=0.0):
    """Backtest a trading strategy based on predictions"""
    try:
        print(colored(f"\n[INFO] Backtesting {model_name}...", "blue"))
        
        # Generate signals
        signals = generate_signals(predictions, threshold)
        
        # Calculate strategy returns
        strategy_returns = signals * actual_returns
        
        # Apply transaction costs (only on position changes)
        position_changes = np.diff(np.concatenate([[0], signals]))
        transaction_costs = np.abs(position_changes) * TRANSACTION_COST
        strategy_returns = strategy_returns - transaction_costs
        
        # Calculate cumulative returns
        cumulative_returns = (1 + strategy_returns).cumprod()
        buy_hold_returns = (1 + actual_returns).cumprod()
        
        # Calculate metrics
        total_return = cumulative_returns.iloc[-1] - 1
        annualized_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
        sharpe_ratio = calculate_sharpe_ratio(strategy_returns)
        max_drawdown = calculate_max_drawdown(cumulative_returns)
        
        # Buy and hold metrics
        bh_total_return = buy_hold_returns.iloc[-1] - 1
        bh_annualized_return = (1 + bh_total_return) ** (252 / len(actual_returns)) - 1
        bh_sharpe_ratio = calculate_sharpe_ratio(actual_returns)
        bh_max_drawdown = calculate_max_drawdown(buy_hold_returns)
        
        # Win rate
        winning_trades = (strategy_returns > 0).sum()
        total_trades = (signals != 0).sum()
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        results = {
            'model_name': model_name,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'buy_hold_return': bh_total_return,
            'buy_hold_annualized': bh_annualized_return,
            'buy_hold_sharpe': bh_sharpe_ratio,
            'buy_hold_max_drawdown': bh_max_drawdown,
            'cumulative_returns': cumulative_returns,
            'buy_hold_cumulative': buy_hold_returns,
            'strategy_returns': strategy_returns
        }
        
        print(colored(f"[SUCCESS] {model_name} backtesting complete", "green"))
        print(colored(f"  Total Return: {total_return*100:.2f}%", "cyan"))
        print(colored(f"  Annualized Return: {annualized_return*100:.2f}%", "cyan"))
        print(colored(f"  Sharpe Ratio: {sharpe_ratio:.4f}", "cyan"))
        print(colored(f"  Max Drawdown: {max_drawdown*100:.2f}%", "cyan"))
        print(colored(f"  Win Rate: {win_rate*100:.2f}%", "cyan"))
        
        return results
        
    except Exception as e:
        print(colored(f"[ERROR] Failed to backtest {model_name}: {str(e)}", "red"))
        raise

def load_predictions():
    """Load ML and DL predictions"""
    try:
        print(colored(f"[INFO] Loading predictions...", "blue"))
        
        ml_preds = pd.read_csv(ML_PREDICTIONS_FILE, index_col=0, parse_dates=True, encoding="utf-8")
        dl_preds = pd.read_csv(DL_PREDICTIONS_FILE, encoding="utf-8")
        
        # Load actual prices to calculate returns
        features = pd.read_csv(FEATURES_FILE, index_col=0, parse_dates=True, encoding="utf-8")
        
        # Calculate actual returns from close prices (if available)
        # Otherwise use target returns from predictions
        if 'close' in features.columns:
            actual_returns = features['close'].pct_change().shift(-1).dropna()
        else:
            # Use actual from ML predictions as proxy
            actual_returns = ml_preds['actual'].values
        
        print(colored(f"[SUCCESS] Predictions loaded", "green"))
        return ml_preds, dl_preds, actual_returns
        
    except Exception as e:
        print(colored(f"[ERROR] Failed to load predictions: {str(e)}", "red"))
        raise

def main():
    """Main function"""
    try:
        # Load predictions
        ml_preds, dl_preds, actual_returns = load_predictions()
        
        # Align data
        common_length = min(len(ml_preds), len(dl_preds), len(actual_returns))
        ml_preds = ml_preds.iloc[:common_length]
        dl_preds = dl_preds.iloc[:common_length]
        actual_returns = actual_returns[:common_length] if isinstance(actual_returns, np.ndarray) else actual_returns.iloc[:common_length]
        
        if isinstance(actual_returns, pd.Series):
            actual_returns = actual_returns.values
        
        results = {}
        
        # Backtest ML models
        print(colored(f"\n{'='*60}", "blue"))
        print(colored("BACKTESTING ML MODELS", "blue", attrs=["bold"]))
        print(colored(f"{'='*60}", "blue"))
        
        for model in ['rf_pred', 'xgb_pred', 'svm_pred']:
            if model in ml_preds.columns:
                model_name = model.replace('_pred', '').upper()
                results[model_name] = backtest_strategy(
                    ml_preds[model].values,
                    actual_returns,
                    f"ML_{model_name}"
                )
        
        # Backtest DL models
        print(colored(f"\n{'='*60}", "blue"))
        print(colored("BACKTESTING DL MODELS", "blue", attrs=["bold"]))
        print(colored(f"{'='*60}", "blue"))
        
        for model in ['transformer_pred', 'cnn_pred', 'lstm_pred']:
            if model in dl_preds.columns:
                model_name = model.replace('_pred', '').replace('cnn', 'DilatedCNN').replace('lstm', 'LSTM')
                results[model_name] = backtest_strategy(
                    dl_preds[model].values,
                    actual_returns,
                    f"DL_{model_name}"
                )
        
        # Save results
        os.makedirs(RESULTS_DIR, exist_ok=True)
        
        # Create summary DataFrame
        summary_data = []
        for model_name, result in results.items():
            summary_data.append({
                'Model': model_name,
                'Total Return (%)': result['total_return'] * 100,
                'Annualized Return (%)': result['annualized_return'] * 100,
                'Sharpe Ratio': result['sharpe_ratio'],
                'Max Drawdown (%)': result['max_drawdown'] * 100,
                'Win Rate (%)': result['win_rate'] * 100,
                'Total Trades': result['total_trades']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = os.path.join(RESULTS_DIR, "backtest_summary.csv")
        summary_df.to_csv(summary_file, index=False, encoding="utf-8")
        print(colored(f"\n[SUCCESS] Backtest summary saved to {summary_file}", "green"))
        
        # Save detailed results
        for model_name, result in results.items():
            detail_df = pd.DataFrame({
                'cumulative_returns': result['cumulative_returns'],
                'buy_hold_cumulative': result['buy_hold_cumulative'],
                'strategy_returns': result['strategy_returns']
            })
            detail_file = os.path.join(RESULTS_DIR, f"{model_name}_backtest_detail.csv")
            detail_df.to_csv(detail_file, encoding="utf-8")
        
        print(colored(f"\n[COMPLETE] Backtesting finished!", "green", attrs=["bold"]))
        
        return results
        
    except Exception as e:
        print(colored(f"[ERROR] Backtesting failed: {str(e)}", "red"))
        raise

if __name__ == "__main__":
    results = main()

