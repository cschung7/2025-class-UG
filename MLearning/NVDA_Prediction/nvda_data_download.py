#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NVDA Data Download Script
Downloads historical NVDA stock data using yfinance
"""

import yfinance as yf
from termcolor import colored
import pandas as pd
import os
from datetime import datetime

# Configuration - ALL CAPS variables
TICKER = "NVDA"
START_DATE = "2010-01-01"
END_DATE = None  # Current date
DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "nvda_raw.csv")

def download_nvda_data():
    """Download NVDA stock data from yfinance"""
    try:
        print(colored(f"[INFO] Starting NVDA data download...", "blue"))
        print(colored(f"[INFO] Ticker: {TICKER}", "cyan"))
        print(colored(f"[INFO] Start Date: {START_DATE}", "cyan"))
        print(colored(f"[INFO] End Date: {END_DATE if END_DATE else 'Current'}", "cyan"))
        
        # Download data
        print(colored(f"[INFO] Downloading data from yfinance...", "yellow"))
        ticker = yf.Ticker(TICKER)
        data = ticker.history(start=START_DATE, end=END_DATE)
        
        if data.empty:
            raise ValueError("Downloaded data is empty")
        
        # Ensure proper column names (lowercase)
        data.columns = [col.lower() for col in data.columns]
        
        # Display basic information
        print(colored(f"\n[SUCCESS] Data downloaded successfully!", "green"))
        print(colored(f"[INFO] Data Shape: {data.shape}", "cyan"))
        print(colored(f"[INFO] Date Range: {data.index[0]} to {data.index[-1]}", "cyan"))
        print(colored(f"[INFO] Columns: {list(data.columns)}", "cyan"))
        
        # Display basic statistics
        print(colored(f"\n[INFO] Basic Statistics:", "blue"))
        print(data.describe())
        
        # Save to CSV
        os.makedirs(DATA_DIR, exist_ok=True)
        data.to_csv(OUTPUT_FILE, encoding="utf-8")
        print(colored(f"\n[SUCCESS] Data saved to: {OUTPUT_FILE}", "green"))
        
        # Display first few rows
        print(colored(f"\n[INFO] First 5 rows:", "blue"))
        print(data.head())
        
        return data
        
    except Exception as e:
        print(colored(f"[ERROR] Failed to download data: {str(e)}", "red"))
        raise

if __name__ == "__main__":
    data = download_nvda_data()
    print(colored(f"\n[COMPLETE] Data download finished!", "green", attrs=["bold"]))

