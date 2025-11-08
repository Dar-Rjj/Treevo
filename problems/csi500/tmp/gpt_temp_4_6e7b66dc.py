import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    """
    Price-Volume Divergence Momentum factor
    Combines price momentum adjusted by volatility with volume trend divergence
    """
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate daily high-low range
    daily_range = df['high'] - df['low']
    
    for i in range(len(df)):
        if i < 4:
            result.iloc[i] = 0
            continue
            
        # Current window indices (t-4 to t)
        window_indices = range(i-4, i+1)
        
        # Price Momentum Component
        close_prices = df['close'].iloc[window_indices].values
        time_periods = np.arange(len(close_prices))
        
        # Calculate price trend slope using linear regression
        if len(set(close_prices)) > 1:  # Ensure we have variation in prices
            price_slope, _, _, _, _ = linregress(time_periods, close_prices)
        else:
            price_slope = 0
        
        # Recent Volatility Adjustment
        recent_volatility = daily_range.iloc[window_indices].mean()
        
        # Avoid division by zero
        if recent_volatility > 0:
            price_momentum = price_slope / recent_volatility
        else:
            price_momentum = price_slope
        
        # Volume Divergence Component
        volumes = df['volume'].iloc[window_indices].values
        
        # Calculate volume trend slope using linear regression
        if len(set(volumes)) > 1:  # Ensure we have variation in volumes
            volume_slope, _, _, _, _ = linregress(time_periods, volumes)
        else:
            volume_slope = 0
        
        # Combine Price Momentum with Volume Trend
        divergence_factor = price_momentum * volume_slope
        
        result.iloc[i] = divergence_factor
    
    return result
