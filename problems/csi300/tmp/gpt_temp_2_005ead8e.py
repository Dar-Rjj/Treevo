import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def heuristics_v2(df, N=20, M=50):
    # Calculate Log Returns
    close_prices = df['close']
    log_returns = np.log(close_prices).diff().dropna()
    
    # Compute Momentum
    momentum = log_returns.rolling(window=N).sum()
    upper_threshold = 0.03
    lower_threshold = -0.03
    momentum = np.clip(momentum, lower_threshold, upper_threshold)
    
    # Adjust for Volume
    volumes = df['volume']
    volume_ratio = volumes / volumes.rolling(window=M).mean()
    adjusted_momentum = momentum * volume_ratio
    
    # Determine Absolute Price Changes
    abs_price_changes = close_prices.diff().abs()
    
    # Calculate Advanced Volatility Measures
    std_dev = abs_price_changes.rolling(window=M).std()
    ema_abs_price_changes = abs_price_changes.ewm(span=M, adjust=False).mean()
    iqr = abs_price_changes.rolling(window=M).quantile(0.75) - abs_price_changes.rolling(window=M).quantile(0.25)
    volatility = (std_dev + ema_abs_price_changes + iqr) / 3
    
    # Integrate Sector-Specific Indicators
    # For simplicity, we assume a single sector for all stocks
    sector_momentum = momentum
    sector_volatility = volatility
    
    # Feature Engineering for Machine Learning
    features = pd.DataFrame({
        'momentum': adjusted_momentum,
        'volatility': volatility,
        'sector_momentum': sector_momentuM,
        'sector_volatility': sector_volatility
    }).dropna()
    
    # Train Model on Historical Data
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    X = features.drop('momentum', axis=1)
    y = features['momentum']
    model.fit(X, y)
    
    # Predict Weights for Each Component
    weights = model.predict(X)
    weighted_momentum = adjusted_momentum * weights
    weighted_volatility = volatility * (1 - weights)
    
    # Final Factor Calculation
    factor = weighted_momentum - weighted_volatility
    return factor

# Example usage
# df = pd.read_csv('stock_data.csv', index_col='date', parse_dates=True)
# factor_values = heuristics_v2(df)
# print(factor_values)
