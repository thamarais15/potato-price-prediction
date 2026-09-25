import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_potato_data(days=1095): # 3 years of data
    np.random.seed(42)
    start_date = datetime(2021, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(days)]
    
    base_price = 20 
    time_index = np.arange(days)
    seasonality = 5 * np.sin(2 * np.pi * time_index / 365)
    trend = 0.005 * time_index
    
    weather_shocks = np.zeros(days)
    shock_indices = np.random.choice(days, size=int(days*0.05), replace=False)
    weather_shocks[shock_indices] = np.random.uniform(3, 8, size=len(shock_indices))
    
    noise = np.random.normal(0, 1, days)
    prices = base_price + seasonality + trend + weather_shocks + noise
    prices = np.maximum(prices, 5) 
    
    temp = 20 + 10 * np.sin(2 * np.pi * time_index / 365) + np.random.normal(0, 2, days)
    rainfall = 50 + 30 * np.cos(2 * np.pi * time_index / 365) + np.random.normal(0, 10, days)
    
    df = pd.DataFrame({
        'date': dates,
        'price': prices,
        'temperature': temp,
        'rainfall': rainfall,
        'demand_index': 1 + 0.2 * np.sin(2 * np.pi * time_index / 365) + np.random.normal(0, 0.05, days)
    })
    
    return df

if __name__ == "__main__":
    df = generate_potato_data()
    # FIX: Changed to relative path for Windows compatibility
    df.to_csv('potato_prices.csv', index=False)
    print("Synthetic data generated and saved to potato_prices.csv")
