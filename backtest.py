import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go

# Calculate the Sharpe ratio
def calculate_sharpe_ratio(returns, risk_free_rate=0.03):
    """
    Calculate the Sharpe ratio given the returns and the risk-free rate.
    """
    excess_returns = returns - risk_free_rate / 252  # Adjust for daily risk-free rate
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)  # Annualized Sharpe ratio

# Calculate the maximum drawdown
def calculate_max_drawdown(cumulative_returns):
    """
    Calculate the maximum drawdown.
    """
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdown = cumulative_returns / running_max - 1
    max_drawdown = np.min(drawdown)
    return max_drawdown

# Generate buy/sell signals based on predicted and actual prices
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

# Evaluate the model and plot predictions vs actual prices
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using common metrics and plot actual vs predicted prices.
    """
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
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
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test_sorted.index, y=y_test_sorted, mode='lines', name='Actual Price', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=y_pred_sorted.index, y=y_pred_sorted, mode='lines', name='Predicted Price', line=dict(color='orange')))
    
    # Update layout
    fig.update_layout(
        title='Actual vs Predicted Stock Prices',
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x unified'
    )
    
    return fig

# Plot the stock prices with buy/sell signals
def plot_with_signals(signals):
    
    """
    Plot stock prices with buy/sell signals using Plotly.
    """
    fig = go.Figure()
    signals = signals.sort_index()

    # Add actual prices
    fig.add_trace(go.Scatter(x=signals.index, y=signals['Actual'], mode='lines', name='Actual Price', line=dict(color='blue')))

    # Add predicted prices
    fig.add_trace(go.Scatter(x=signals.index, y=signals['Predicted'], mode='lines', name='Predicted Price', line=dict(color='orange')))

    # Add buy signals as green triangles
    buy_signals = signals[signals['Signal'] == 1]
    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Actual'], mode='markers', name='Buy Signal', marker=dict(symbol='triangle-up', color='green', size=10)))

    # Add sell signals as red triangles
    sell_signals = signals[signals['Signal'] == -1]
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Actual'], mode='markers', name='Sell Signal', marker=dict(symbol='triangle-down', color='red', size=10)))

    # Update layout
    fig.update_layout(
        title='Stock Price with Buy/Sell Signals',
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x',
        xaxis_rangeslider_visible=True  # Adds a range slider for zooming in/out
    )

    return fig

def backtest_strategy(signals, initial_balance=10000):
    # Fixing the buy-and-hold strategy first
    # Starting with the initial balance
    first_price = signals['Actual'].iloc[0]
    shares_bought = initial_balance / first_price  # Buy as many shares as possible at the start
    buy_and_hold_final_price = signals['Actual'].iloc[-1]  # Price at the end of the period
    buy_and_hold_final_balance = shares_bought * buy_and_hold_final_price  # Final value of holding those shares
    buy_and_hold_return = (buy_and_hold_final_balance - initial_balance) / initial_balance * 100
    
    # Debugging the buy-and-hold strategy
    print(f"Buy-and-Hold: Bought at {first_price}, Sold at {buy_and_hold_final_price}")
    print(f"Shares bought: {shares_bought}, Final balance: {buy_and_hold_final_balance}, Return: {buy_and_hold_return}%")

    # Trading strategy variables
    balance = initial_balance
    position = 0  # Track current position (number of shares)

    for i, row in signals.iterrows():
        price = row['Actual']
        signal = row['Signal']

        # Trading Strategy Logic
        if signal == 1 and position == 0:
            # Buy as many shares as possible with the available balance
            shares_to_buy = balance // price  # Floor division
            if shares_to_buy > 0:
                position = shares_to_buy
                balance -= shares_to_buy * price  # Deduct from cash balance
        
        elif signal == -1 and position > 0:
            # Sell all shares held
            balance += position * price  # Liquidate position
            position = 0  # Reset position after selling

    # Liquidate any remaining position at the last price for trading strategy
    if position > 0:
        balance += position * signals['Actual'].iloc[-1]
        position = 0

    # Final balance calculation for trading strategy
    final_balance = balance
    total_return = (final_balance - initial_balance) / initial_balance * 100

    # Debugging the trading strategy
    print(f"Trading Strategy: Final balance: {final_balance}, Return: {total_return}%")

    return final_balance, total_return, buy_and_hold_final_balance, buy_and_hold_return


def backtest_results_df(signals, initial_balance=1000):

    # Run the backtest
    final_balance, total_return, buy_and_hold_balance, buy_and_hold_return = backtest_strategy(signals, initial_balance)

    # Create a DataFrame to store the results
    results = pd.DataFrame({
        'Strategy': ['Trading Strategy', 'Buy-and-Hold Strategy'],
        'Final Balance': [round(final_balance, 2), round(buy_and_hold_balance, 2)],
        'Total Return (%)': [round(total_return, 2), round(buy_and_hold_return, 2)]
    })

    return results

