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
