import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json

def create_deep_features(df):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # 1. Time-based features
    df['day_of_year'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    
    # 2. Extensive Lag features
    for lag in [1, 2, 3, 7, 14, 21, 30, 60]:
        df[f'price_lag_{lag}'] = df['price'].shift(lag)
        
    # 3. Advanced Rolling Statistics
    for window in [7, 14, 30]:
        df[f'rolling_mean_{window}'] = df['price'].shift(1).rolling(window=window).mean()
        df[f'rolling_std_{window}'] = df['price'].shift(1).rolling(window=window).std()
        df[f'rolling_max_{window}'] = df['price'].shift(1).rolling(window=window).max()
    
    # 4. Change rates (Momentum)
    df['price_change_1d'] = df['price'].shift(1) - df['price'].shift(2)
    
    # 5. Complex Interactions
    df['weather_impact'] = df['temperature'] * df['rainfall']
    df['demand_volatility'] = df['demand_index'] * df['price'].shift(1)
    
    return df.dropna()

def train_and_evaluate(model_name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return {
        'MAE': mean_absolute_error(y_test, preds),
        'RMSE': np.sqrt(mean_squared_error(y_test, preds)),
        'R2': r2_score(y_test, preds)
    }

if __name__ == "__main__":
    try:
        df = pd.read_csv('potato_prices.csv')
    except FileNotFoundError:
        print("Data file not found. Please run data_gen.py first.")
        exit()

    df_featured = create_deep_features(df)
    
    features = [col for col in df_featured.columns if col not in ['date', 'price']]
    X = df_featured[features]
    y = df_featured['price']
    
    # Time-series split (no shuffle)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # Expanded Model Suite for Maximum Accuracy
    models = {
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'ElasticNet': ElasticNet(),
        'RandomForest': RandomForestRegressor(n_estimators=200, random_state=42),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=200, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=200, random_state=42),
        'AdaBoost': AdaBoostRegressor(n_estimators=200, random_state=42),
        'XGBoost': XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42)
    }
    
    all_results = {}
    for name, model in models.items():
        metrics = train_and_evaluate(name, model, X_train, X_test, y_train, y_test)
        all_results[name] = metrics
    
    with open('model_performance.json', 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print("High-accuracy model evaluation complete. Results saved to model_performance.json")
