import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress, pearsonr

def heuristics_v2(df):
    """
    Price-Momentum and Volume Divergence Factor
    Combines short-term and medium-term price-volume relationships
    """
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate components with sufficient lookback
    for i in range(len(df)):
        if i < 20:  # Need at least 20 days for all calculations
            result.iloc[i] = 0
            continue
            
        current_data = df.iloc[:i+1]  # Only use current and past data
        
        # Short-Term Price-Volume Momentum (6-day)
        if i >= 5:
            # 6-day Linear Regression Slope of Close prices
            close_window = current_data['close'].iloc[-6:]
            x_close = np.arange(len(close_window))
            close_slope = linregress(x_close, close_window.values).slope
            
            # 6-day Linear Regression Slope of Volume
            volume_window = current_data['volume'].iloc[-6:]
            x_volume = np.arange(len(volume_window))
            volume_slope = linregress(x_volume, volume_window.values).slope
            
            short_term_component = close_slope * volume_slope
        else:
            short_term_component = 0
        
        # Medium-Term Price-Volume Relationship (20-day)
        # 20-day Linear Regression Slope of Close prices
        close_window_20 = current_data['close'].iloc[-20:]
        x_close_20 = np.arange(len(close_window_20))
        close_slope_20 = linregress(x_close_20, close_window_20.values).slope
        
        # 10-day Pearson Correlation between Close and Volume
        if i >= 9:
            close_corr_window = current_data['close'].iloc[-10:]
            volume_corr_window = current_data['volume'].iloc[-10:]
            price_volume_corr = pearsonr(close_corr_window.values, volume_corr_window.values)[0]
        else:
            price_volume_corr = 0
        
        medium_term_component = close_slope_20 * price_volume_corr
        
        # Combine components
        result.iloc[i] = short_term_component + medium_term_component
    
    return result
