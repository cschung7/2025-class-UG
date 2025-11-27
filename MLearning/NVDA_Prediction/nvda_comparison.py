#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NVDA Comparison Script
Compares ML and DL model performance with visualizations
"""

import sys
import os
import pandas as pd
import numpy as np
from termcolor import colored
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('dark_background')
sns.set_palette("husl")

# Configuration - ALL CAPS variables
RESULTS_DIR = "results"
BACKTEST_DIR = os.path.join(RESULTS_DIR, "backtest_results")
VISUALIZATIONS_DIR = os.path.join(RESULTS_DIR, "visualizations")
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, "predictions")

BACKTEST_SUMMARY_FILE = os.path.join(BACKTEST_DIR, "backtest_summary.csv")
ML_PREDICTIONS_FILE = os.path.join(PREDICTIONS_DIR, "ml_predictions.csv")
DL_PREDICTIONS_FILE = os.path.join(PREDICTIONS_DIR, "dl_predictions.csv")

def load_results():
    """Load all results"""
    try:
        print(colored(f"[INFO] Loading results...", "blue"))
        
        summary = pd.read_csv(BACKTEST_SUMMARY_FILE, encoding="utf-8")
        ml_preds = pd.read_csv(ML_PREDICTIONS_FILE, index_col=0, parse_dates=True, encoding="utf-8")
        dl_preds = pd.read_csv(DL_PREDICTIONS_FILE, encoding="utf-8")
        
        print(colored(f"[SUCCESS] Results loaded", "green"))
        return summary, ml_preds, dl_preds
        
    except Exception as e:
        print(colored(f"[ERROR] Failed to load results: {str(e)}", "red"))
        raise

def create_comparison_table(summary):
    """Create comprehensive comparison table"""
    try:
        print(colored(f"\n[INFO] Creating comparison table...", "blue"))
        
        # Sort by Sharpe ratio
        summary_sorted = summary.sort_values('Sharpe Ratio', ascending=False)
        
        print(colored(f"\n{'='*80}", "blue"))
        print(colored("MODEL PERFORMANCE COMPARISON", "blue", attrs=["bold"]))
        print(colored(f"{'='*80}", "blue"))
        print(summary_sorted.to_string(index=False))
        print(colored(f"{'='*80}", "blue"))
        
        return summary_sorted
        
    except Exception as e:
        print(colored(f"[ERROR] Failed to create comparison table: {str(e)}", "red"))
        raise

def plot_performance_metrics(summary):
    """Plot performance metrics comparison"""
    try:
        print(colored(f"[INFO] Creating performance metrics plots...", "blue"))
        
        os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', color='white')
        
        # Sharpe Ratio
        ax1 = axes[0, 0]
        summary_sorted = summary.sort_values('Sharpe Ratio', ascending=True)
        ax1.barh(summary_sorted['Model'], summary_sorted['Sharpe Ratio'], color='cyan')
        ax1.set_xlabel('Sharpe Ratio', color='white')
        ax1.set_title('Sharpe Ratio Comparison', color='white', fontweight='bold')
        ax1.tick_params(colors='white')
        ax1.grid(True, alpha=0.3)
        
        # Annualized Return
        ax2 = axes[0, 1]
        summary_sorted = summary.sort_values('Annualized Return (%)', ascending=True)
        ax2.barh(summary_sorted['Model'], summary_sorted['Annualized Return (%)'], color='lime')
        ax2.set_xlabel('Annualized Return (%)', color='white')
        ax2.set_title('Annualized Return Comparison', color='white', fontweight='bold')
        ax2.tick_params(colors='white')
        ax2.grid(True, alpha=0.3)
        
        # Max Drawdown
        ax3 = axes[1, 0]
        summary_sorted = summary.sort_values('Max Drawdown (%)', ascending=False)
        ax3.barh(summary_sorted['Model'], summary_sorted['Max Drawdown (%)'], color='red')
        ax3.set_xlabel('Max Drawdown (%)', color='white')
        ax3.set_title('Max Drawdown Comparison', color='white', fontweight='bold')
        ax3.tick_params(colors='white')
        ax3.grid(True, alpha=0.3)
        
        # Win Rate
        ax4 = axes[1, 1]
        summary_sorted = summary.sort_values('Win Rate (%)', ascending=True)
        ax4.barh(summary_sorted['Model'], summary_sorted['Win Rate (%)'], color='yellow')
        ax4.set_xlabel('Win Rate (%)', color='white')
        ax4.set_title('Win Rate Comparison', color='white', fontweight='bold')
        ax4.tick_params(colors='white')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = os.path.join(VISUALIZATIONS_DIR, "performance_metrics.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='black')
        print(colored(f"[SUCCESS] Performance metrics plot saved to {plot_file}", "green"))
        plt.close()
        
    except Exception as e:
        print(colored(f"[ERROR] Failed to create performance plots: {str(e)}", "red"))
        raise

def plot_equity_curves():
    """Plot equity curves for all models"""
    try:
        print(colored(f"[INFO] Creating equity curves...", "blue"))
        
        # Load detailed backtest results
        equity_data = {}
        for file in os.listdir(BACKTEST_DIR):
            if file.endswith('_backtest_detail.csv'):
                model_name = file.replace('_backtest_detail.csv', '')
                df = pd.read_csv(os.path.join(BACKTEST_DIR, file), encoding="utf-8")
                equity_data[model_name] = df['cumulative_returns']
        
        if not equity_data:
            print(colored(f"[WARNING] No equity curve data found", "yellow"))
            return
        
        plt.figure(figsize=(14, 8))
        
        for model_name, equity in equity_data.items():
            plt.plot(equity.values, label=model_name, linewidth=2)
        
        plt.xlabel('Time', color='white', fontsize=12)
        plt.ylabel('Cumulative Returns', color='white', fontsize=12)
        plt.title('Equity Curves Comparison', color='white', fontsize=14, fontweight='bold')
        plt.legend(loc='best', facecolor='black', edgecolor='white', labelcolor='white')
        plt.grid(True, alpha=0.3)
        plt.tick_params(colors='white')
        
        plot_file = os.path.join(VISUALIZATIONS_DIR, "equity_curves.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='black')
        print(colored(f"[SUCCESS] Equity curves plot saved to {plot_file}", "green"))
        plt.close()
        
    except Exception as e:
        print(colored(f"[ERROR] Failed to create equity curves: {str(e)}", "red"))
        raise

def plot_predictions_vs_actual(ml_preds, dl_preds):
    """Plot predictions vs actual values"""
    try:
        print(colored(f"[INFO] Creating predictions vs actual plots...", "blue"))
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Predictions vs Actual Returns', fontsize=16, fontweight='bold', color='white')
        
        # ML Models
        ml_models = ['rf_pred', 'xgb_pred', 'svm_pred']
        for idx, model in enumerate(ml_models):
            if model in ml_preds.columns:
                ax = axes[0, idx]
                actual = ml_preds['actual'].values
                pred = ml_preds[model].values
                
                # Scatter plot
                ax.scatter(actual, pred, alpha=0.5, s=10, color='cyan')
                ax.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
                ax.set_xlabel('Actual Returns', color='white')
                ax.set_ylabel('Predicted Returns', color='white')
                ax.set_title(f'{model.replace("_pred", "").upper()}', color='white', fontweight='bold')
                ax.tick_params(colors='white')
                ax.grid(True, alpha=0.3)
        
        # DL Models
        dl_models = ['transformer_pred', 'cnn_pred', 'lstm_pred']
        for idx, model in enumerate(dl_models):
            if model in dl_preds.columns:
                ax = axes[1, idx]
                actual = dl_preds['actual'].values if 'actual' in dl_preds.columns else ml_preds['actual'].values[:len(dl_preds)]
                pred = dl_preds[model].values
                
                # Scatter plot
                ax.scatter(actual[:len(pred)], pred, alpha=0.5, s=10, color='lime')
                min_val = min(actual[:len(pred)].min(), pred.min())
                max_val = max(actual[:len(pred)].max(), pred.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
                ax.set_xlabel('Actual Returns', color='white')
                ax.set_ylabel('Predicted Returns', color='white')
                model_name = model.replace('_pred', '').replace('cnn', 'DilatedCNN').replace('lstm', 'LSTM')
                ax.set_title(f'{model_name}', color='white', fontweight='bold')
                ax.tick_params(colors='white')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = os.path.join(VISUALIZATIONS_DIR, "predictions_vs_actual.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='black')
        print(colored(f"[SUCCESS] Predictions vs actual plot saved to {plot_file}", "green"))
        plt.close()
        
    except Exception as e:
        print(colored(f"[ERROR] Failed to create predictions plot: {str(e)}", "red"))
        raise

def generate_summary_report(summary):
    """Generate text summary report"""
    try:
        print(colored(f"\n[INFO] Generating summary report...", "blue"))
        
        report_file = os.path.join(RESULTS_DIR, "comparison_report.txt")
        
        with open(report_file, 'w', encoding="utf-8") as f:
            f.write("="*80 + "\n")
            f.write("NVDA STOCK PREDICTION - ML vs DL COMPARISON REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-"*80 + "\n")
            best_sharpe = summary.loc[summary['Sharpe Ratio'].idxmax()]
            f.write(f"Best Model (by Sharpe Ratio): {best_sharpe['Model']}\n")
            f.write(f"  - Sharpe Ratio: {best_sharpe['Sharpe Ratio']:.4f}\n")
            f.write(f"  - Annualized Return: {best_sharpe['Annualized Return (%)']:.2f}%\n")
            f.write(f"  - Max Drawdown: {best_sharpe['Max Drawdown (%)']:.2f}%\n\n")
            
            f.write("DETAILED RESULTS\n")
            f.write("-"*80 + "\n")
            f.write(summary.to_string(index=False))
            f.write("\n\n")
            
            f.write("KEY FINDINGS\n")
            f.write("-"*80 + "\n")
            f.write(f"1. Total models evaluated: {len(summary)}\n")
            f.write(f"2. Best Sharpe Ratio: {summary['Sharpe Ratio'].max():.4f}\n")
            f.write(f"3. Best Annualized Return: {summary['Annualized Return (%)'].max():.2f}%\n")
            f.write(f"4. Lowest Max Drawdown: {summary['Max Drawdown (%)'].min():.2f}%\n")
            f.write(f"5. Best Win Rate: {summary['Win Rate (%)'].max():.2f}%\n")
        
        print(colored(f"[SUCCESS] Summary report saved to {report_file}", "green"))
        
    except Exception as e:
        print(colored(f"[ERROR] Failed to generate report: {str(e)}", "red"))
        raise

def main():
    """Main function"""
    try:
        # Load results
        summary, ml_preds, dl_preds = load_results()
        
        # Create comparison table
        summary_sorted = create_comparison_table(summary)
        
        # Create visualizations
        plot_performance_metrics(summary_sorted)
        plot_equity_curves()
        plot_predictions_vs_actual(ml_preds, dl_preds)
        
        # Generate report
        generate_summary_report(summary_sorted)
        
        print(colored(f"\n[COMPLETE] Comparison analysis finished!", "green", attrs=["bold"]))
        print(colored(f"\nResults saved in: {RESULTS_DIR}", "cyan"))
        print(colored(f"Visualizations saved in: {VISUALIZATIONS_DIR}", "cyan"))
        
        return summary_sorted
        
    except Exception as e:
        print(colored(f"[ERROR] Comparison failed: {str(e)}", "red"))
        raise

if __name__ == "__main__":
    summary = main()

