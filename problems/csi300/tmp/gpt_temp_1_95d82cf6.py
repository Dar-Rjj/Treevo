import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Calculate daily returns
    returns = df['close'].pct_change()
    
    # Compute rolling window returns (20-day)
    rolling_returns = returns.rolling(window=20)
    
    # Calculate rolling volatility (20-day standard deviation of returns)
    volatility = rolling_returns.std()
    
    # Calculate price momentum (5-day price change)
    price_change = df['close'] - df['close'].shift(5)
    
    # Calculate momentum ratio (price change divided by volatility)
    # Add small epsilon to avoid division by zero
    epsilon = 1e-8
    momentum_ratio = price_change / (volatility + epsilon)
    
    # Scale by average volatility (20-day rolling average of volatility)
    avg_volatility = volatility.rolling(window=20).mean()
    scaled_momentum = momentum_ratio / (avg_volatility + epsilon)
    
    # Calculate volume trend using linear regression slope (10-day window)
    def volume_slope(volume_series):
        if len(volume_series) < 2:
            return np.nan
        x = np.arange(len(volume_series))
        slope, _, _, _, _ = linregress(x, volume_series)
        return slope
    
    volume_trend = df['volume'].rolling(window=10).apply(
        volume_slope, raw=False
    )
    
    # Normalize volume trend to make it more comparable across stocks
    volume_trend_normalized = volume_trend / (df['volume'].rolling(window=10).mean() + epsilon)
    
    # Combine momentum with volume confirmation
    # Multiply momentum by volume trend and apply sign adjustment
    factor = scaled_momentum * volume_trend_normalized
    
    # Apply sign adjustment based on direction (positive momentum with positive volume trend gets amplified)
    # This is already achieved through multiplication
    
    return factor
