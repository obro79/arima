from data_preparation import download_stock_data, load_stock_data
from feature_engineering import create_features
from train_model import train_xgboost_model
from evaluate import evaluate_model
from save_load_data import save_model, load_model

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
y = data_with_features['Close']

# Step 5: Train the XGBoost model
model, X_train, X_test, y_train, y_test = train_xgboost_model(X, y)

# Step 6: Save the trained model
save_model(model)

# Step 7: Evaluate the model's performance
evaluate_model(model, X_test, y_test)
