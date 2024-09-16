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
    
import pandas as pd

def create_features(data):
    """Create lagged features and technical indicators."""
    # Create lagged features
    data['Price_Lag_1'] = data['Adj Close'].shift(1)
    data['Price_Lag_2'] = data['Adj Close'].shift(2)
    data['Price_Lag_3'] = data['Adj Close'].shift(3)
    
    # Create moving averages
    data['MA_50'] = data['Adj Close'].rolling(window=50).mean()
    data['MA_200'] = data['Adj Close'].rolling(window=200).mean()
    
    # Create more features if needed (e.g., Volume, RSI)
    data['Volume'] = data['Volume']
    delta = data['Adj Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Create Bollinger Bands
    data['Bollinger_High'] = data['MA_50'] + (data['Adj Close'].rolling(window=50).std() * 2)
    data['Bollinger_Low'] = data['MA_50'] - (data['Adj Close'].rolling(window=50).std() * 2)
    
    # Drop NaN values caused by shifting/rolling
    data = data.dropna()
    
    return data
    
