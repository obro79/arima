import yfinance as yf
import pandas as pd
import os

def download_stock_data(ticker, start, end, save_path="data.csv"):
    """Download stock data from Yahoo Finance and save it to a CSV file."""
    if not os.path.exists(save_path):
        data = yf.download(ticker, start=start, end=end)
        data.to_csv(save_path)
        print(f"Data downloaded and saved to {save_path}")
    else:
        print(f"Data already exists at {save_path}")

def load_stock_data(file_path="data.csv"):
    """Load stock data from a CSV file."""
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        raise FileNotFoundError(f"No data file found at {file_path}. Please download the data first.")
