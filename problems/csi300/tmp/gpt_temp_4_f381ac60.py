import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Intraday Volatility
    intraday_volatility = df['high'] - df['low']
    
    # Weight by Volume
    weighted_volatility = intraday_volatility * df['volume']
    
    # Calculate Intraday Momentum
    daily_momentum = df['close'] - df['open']
    rolling_momentum = daily_momentum.rolling(window=5).sum()
    
    # Integrate Momentum and Volatility
    integrated_value = weighted_volatility + rolling_momentum
    
    # Apply Dynamic Exponential Smoothing
    smoothing_factor = 0.9
    smoothed_value = integrated_value.ewm(alpha=1 - smoothing_factor, adjust=False).mean()
    
    # Ensure Values are Positive
    smoothed_value = smoothed_value + 1e-6  # Add a small constant to avoid log of zero or negative
    
    # Adjust for Market Conditions
    close_ma = df['close'].rolling(window=20).mean()
    close_std = df['close'].rolling(window=20).std()
    
    # Define market trend based on moving average
    is_trending_up = df['close'] > close_ma
    adjusted_smoothing_factor = np.where(is_trending_up, 0.95, 0.85)
    
    # Reapply Exponential Smoothing with Adjusted Factor
    final_smoothed_value = smoothed_value.ewm(alpha=1 - adjusted_smoothing_factor, adjust=False).mean()
    
    # Apply Logarithmic Transformation
    log_transformed_value = np.log(final_smoothed_value)
    
    # Factor Output
    return log_transformed_value
