
def backtest_strategy(signals, initial_balance=10000):
    """
    Simulate a simple backtest of the buy/sell signals.
    initial_balance: Starting balance for the strategy
    Returns the final balance and performance metrics.
    """
    balance = initial_balance
    position = 0  # 0 means no stock, 1 means holding stock
    shares = 0  # Number of shares bought
    buy_price = 0
    
    for i in range(1, len(signals)):
        # If we get a buy signal and we're not holding stock
        if signals['Signal'].iloc[i] == 1 and position == 0:
            shares = balance // signals['Actual'].iloc[i]  # Buy as many shares as possible
            buy_price = signals['Actual'].iloc[i]
            balance -= shares * buy_price
            position = 1  # We are now holding stock
        
        # If we get a sell signal and we're holding stock
        elif signals['Signal'].iloc[i] == -1 and position == 1:
            sell_price = signals['Actual'].iloc[i]
            balance += shares * sell_price
            shares = 0
            position = 0  # Sold all shares
    
    # If holding stock at the end, sell at the last price
    if position == 1:
        balance += shares * signals['Actual'].iloc[-1]
    
    # Calculate total return and other performance metrics
    total_return = (balance - initial_balance) / initial_balance * 100
    
    print(f"Initial Balance: ${initial_balance}")
    print(f"Final Balance: ${balance}")
    print(f"Total Return: {total_return:.2f}%")
    
    return balance, total_return

def buy_and_hold_strategy(signals, initial_balance=10000):
    """
    Simulate a buy-and-hold strategy where you buy the stock at the first trading day and hold it until the last day.
    """
    balance = initial_balance
    initial_price = signals['Actual'].iloc[0]
    final_price = signals['Actual'].iloc[-1]
    
    # Buy as many shares as possible on the first day
    shares = balance // initial_price
    balance = balance - shares * initial_price  # Update balance after buying
    
    # At the end of the period, sell all shares
    balance += shares * final_price
    
    # Calculate total return
    total_return = (balance - initial_balance) / initial_balance * 100
    print(f"Buy and Hold Initial Balance: ${initial_balance}")
    print(f"Buy and Hold Final Balance: ${balance}")
    print(f"Buy and Hold Total Return: {total_return:.2f}%")
    return balance, total_return
