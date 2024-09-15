#evaluate_model.py
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and plot actual vs predicted prices."""
    
    # Predict stock prices
    y_pred = model.predict(X_test)
    
    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"R² Score: {r2}")
    

    # Restore the original indices for the test set
    y_test_sorted = y_test.sort_index()
    y_pred_sorted = pd.Series(y_pred, index=y_test.index).sort_index()
    
    # Plot actual vs predicted prices
    plt.figure(figsize=(10, 6))
    plt.plot(y_test_sorted.index, y_test_sorted, label='Actual Price', color='blue')
    plt.plot(y_pred_sorted.index, y_pred_sorted, label='Predicted Price', color='orange')
    plt.title('Actual vs Predicted Stock Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

# Inside evaluate_model.py (or wherever you handle evaluation)
import pandas as pd

def generate_signals(y_test, y_pred, threshold=0.02):
    """
    Generate buy/sell signals based on predicted and actual prices with a percentage threshold.
    threshold: The minimum percentage difference between predicted and actual prices for a signal.
    A '1' indicates a buy signal, and '-1' indicates a sell signal.
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
