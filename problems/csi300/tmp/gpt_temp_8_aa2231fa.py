import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    # Compute 5-day price acceleration
    # Calculate 3-day return using Close price
    ret_3d = df['close'].pct_change(3)
    
    # Calculate 5-day return using Close price
    ret_5d = df['close'].pct_change(5)
    
    # Subtract 3-day return from 5-day return
    price_acceleration = ret_5d - ret_3d
    
    # Calculate volume trend strength
    def rolling_slope(series, window):
        slopes = pd.Series(index=series.index, dtype=float)
        for i in range(window-1, len(series)):
            if i >= window-1:
                y = series.iloc[i-window+1:i+1].values
                x = np.arange(window)
                slope, _, _, _, _ = stats.linregress(x, y)
                slopes.iloc[i] = slope
        return slopes
    
    # Compute 5-day volume slope using linear regression
    volume_slope_5d = rolling_slope(df['volume'], 5)
    
    # Compute 20-day volume slope using linear regression
    volume_slope_20d = rolling_slope(df['volume'], 20)
    
    # Take ratio of 5-day to 20-day volume slopes
    volume_trend_ratio = volume_slope_5d / volume_slope_20d
    
    # Combine price acceleration with volume trend
    # Multiply price acceleration by volume trend ratio
    combined_factor = price_acceleration * volume_trend_ratio
    
    # Apply sign function to preserve direction
    # Multiply by absolute value of 3-day return for magnitude scaling
    final_factor = np.sign(combined_factor) * np.abs(ret_3d)
    
    return final_factor
