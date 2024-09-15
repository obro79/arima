import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
from config import model_params

def train_xgboost_model(X, y):
    """Train the XGBoost model using the provided features and target."""
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize the XGBoost model with hyperparameters from config.py
    model = xgb.XGBRegressor(
        n_estimators=model_params['n_estimators'],
        learning_rate=model_params['learning_rate'],
        max_depth=model_params['max_depth'],
        random_state=42
    )
    
    mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring=mse_scorer)
    
    print(f"Mean Squared Error for each fold: {cv_scores}")
    print(f"Average MSE across all folds: {abs(cv_scores.mean())}")
    
    # Train the model
    model.fit(X_train, y_train)
    
    return model, X_train, X_test, y_train, y_test
