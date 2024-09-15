# trading_signals.py
import pandas as pd

def generate_signals(y_test, y_pred, threshold=0.02):
    """
    Generate buy/sell signals based on predicted and actual prices with a percentage threshold.
    """
    signals = pd.DataFrame(index=y_test.index)
    signals['Actual'] = y_test
    signals['Predicted'] = y_pred
    
    # Calculate percentage difference between predicted and actual prices
    signals['Difference'] = (signals['Predicted'] - signals['Actual']) / signals['Actual']
    
    # Initialize signal column to zero (no signal)
    signals['Signal'] = 0
    
    # Buy when predicted price is greater than actual price by more than the threshold
    signals.loc[signals['Difference'] > threshold, 'Signal'] = 1  # Buy signal
    
    # Sell when predicted price is lower than actual price by more than the threshold
    signals.loc[signals['Difference'] < -threshold, 'Signal'] = -1  # Sell signal
    
    return signals
