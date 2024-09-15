import joblib

def save_model(model, file_path="/Users/owenfisher/Desktop/Trading Strat/arima/xgboost_model_200e.pkl"):
    """Save the trained model to a file."""
    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}")

def load_model(file_path="xgboost_model.pkl"):
    """Load the model from a file."""
    return joblib.load(file_path)
