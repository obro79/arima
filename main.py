from data_preparation import download_stock_data, load_stock_data, create_features
from train_model import train_xgboost_model
from save_load_data import save_model, load_model
import matplotlib.pyplot as plt
from backtest import backtest_strategy, buy_and_hold_strategy, backtest_with_metrics, generate_signals, evaluate_model , plot_with_signals 

## source venv
# Set the stock symbol and date range
ticker = 'AAPL'
start_date = '2000-01-01'
end_date = '2023-01-01' 

# Step 1: Download the stock data (if not already downloaded)
download_stock_data(ticker, start=start_date, end=end_date)

# Step 2: Load the stock data
data = load_stock_data()

# Step 3: Create features for XGBoost
data_with_features = create_features(data)

# Step 4: Define the features (X) and target (y)
X = data_with_features[['Price_Lag_1', 'Price_Lag_2', 'Price_Lag_3', 'MA_50', 'MA_200', 'Volume', 'RSI', 'Bollinger_High', 'Bollinger_Low']]
y = data_with_features['Adj Close']

# Step 5: Train the XGBoost model
model, X_train, X_test, y_train, y_test = train_xgboost_model(X, y)

# Step 6: Save the trained model
save_model(model)

# Step 7: Evaluate the model's performance
evaluate_model(model, X_test, y_test)

# Step 8: Generate buy/sell signals based on a 2% threshold
y_pred = model.predict(X)  
signals = generate_signals(y, y_pred, threshold=0.02)  # Use 2% threshold for signals

# Step 9: Backtest the strategy
final_balance, total_return = backtest_strategy(signals)

print(backtest_with_metrics(signals))

plt.figure(figsize=(10, 6))
plt.plot(signals.index, signals['Actual'], label='Actual Price')
plt.plot(signals.index, signals['Predicted'], label='Predicted Price', color='orange')

# Mark Buy and Sell signals
buy_signals = signals[signals['Signal'] == 1]
sell_signals = signals[signals['Signal'] == -1]
plt.scatter(buy_signals.index, buy_signals['Actual'], marker='^', color='green', label='Buy Signal', alpha=1)
plt.scatter(sell_signals.index, sell_signals['Actual'], marker='v', color='red', label='Sell Signal', alpha=1)

plt.title('Stock Price with Buy/Sell Signals')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

buy_and_hold_final_balance, buy_and_hold_total_return = buy_and_hold_strategy(signals)

# Print a comparison
print(f"Buy/Sell Strategy Total Return: {total_return:.2f}%")
print(f"Buy and Hold Strategy Total Return: {buy_and_hold_total_return:.2f}%")