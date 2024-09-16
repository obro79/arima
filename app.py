import streamlit as st
import yfinance as yf
from data_preparation import download_stock_data, load_stock_data, create_features
from train_model import train_xgboost_model
from backtest import backtest_strategy, generate_signals, evaluate_model, plot_with_signals, backtest_results_df
import matplotlib.pyplot as plt
from datetime import datetime

# App title
st.title("Stock XGBoost Model")

# Step 1: User Input - Only ticker and date range are adjustable
st.sidebar.header("Select Stock and Date Range")
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=datetime(2020, 1, 1))
end_date = st.sidebar.date_input("End Date", value=datetime(2023, 1, 1))

# Fixed parameters (not adjustable by user)
threshold = 0.02  # Fixed buy/sell signal threshold

# Step 2: Load Stock Data
st.header(f"Stock Data for {ticker}")
data = yf.download(ticker, start=start_date, end=end_date)
st.line_chart(data['Adj Close'])

# Step 3: Feature Engineering
data_with_features = create_features(data)

# Step 4: Train the Model
X = data_with_features[['Price_Lag_1', 'Price_Lag_2', 'Price_Lag_3', 'MA_50', 'MA_200', 'RSI', 'Bollinger_High', 'Bollinger_Low']]
y = data_with_features['Adj Close']
model, X_train, X_test, y_train, y_test = train_xgboost_model(X, y)

# Step 5: Generate Buy/Sell Signals
y_pred = model.predict(X_test)
signals = generate_signals(y_test, y_pred, threshold=threshold)

# Step 6: Run Backtest
final_balance, total_return, _ , _ = backtest_strategy(signals)

st.title("Predicted vs Actual")
st.plotly_chart(evaluate_model(model,X_test,y_test))

# Step 7: Plot Signals
st.header("Buy/Sell Signals")
st.plotly_chart(plot_with_signals(signals))
