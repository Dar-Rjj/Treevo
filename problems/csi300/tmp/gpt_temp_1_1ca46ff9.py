import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(data):
    df = data.copy()
    
    # Calculate 5-day Rate of Change for Close prices
    roc_5 = df['close'] / df['close'].shift(5) - 1
    
    # Calculate Price Acceleration (ROC change)
    # ROC from t-5 to t minus ROC from t-10 to t-5
    roc_prev = df['close'].shift(5) / df['close'].shift(10) - 1
    price_acceleration = roc_5 - roc_prev
    
    # Calculate Volume Trend using linear regression slope (10-day window)
    def volume_slope(volume_series):
        if len(volume_series) < 10:
            return np.nan
        x = np.arange(10)
        y = volume_series.values
        slope, _, _, _, _ = stats.linregress(x, y)
        return slope
    
    volume_trend = df['volume'].rolling(window=10, min_periods=10).apply(volume_slope, raw=False)
    
    # Calculate Price Trend using linear regression slope (10-day window)
    def price_slope(price_series):
        if len(price_series) < 10:
            return np.nan
        x = np.arange(10)
        y = price_series.values
        slope, _, _, _, _ = stats.linregress(x, y)
        return slope
    
    price_trend = df['close'].rolling(window=10, min_periods=10).apply(price_slope, raw=False)
    
    # Compute Volume Divergence Ratio
    # Avoid division by zero and handle cases where trends are very small
    volume_divergence = volume_trend / (abs(price_trend) + 1e-8)
    
    # Combine Acceleration and Divergence
    combined_factor = price_acceleration * volume_divergence
    
    # Apply Sign Correction using direction of most recent price change
    recent_price_change = df['close'] - df['close'].shift(1)
    sign_correction = np.sign(recent_price_change)
    
    # Final factor with sign correction
    final_factor = combined_factor * sign_correction
    
    return final_factor
