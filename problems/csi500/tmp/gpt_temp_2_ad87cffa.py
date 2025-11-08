import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(data):
    """
    Price-Volume Divergence Momentum factor that compares price momentum 
    against volume momentum across different time horizons.
    """
    close = data['close']
    volume = data['volume']
    
    # Initialize result series
    factor_values = pd.Series(index=close.index, dtype=float)
    
    for i in range(20, len(close)):
        current_date = close.index[i]
        
        # Recent period (t-5 to t)
        recent_close = close.iloc[i-5:i+1].values
        recent_volume = volume.iloc[i-5:i+1].values
        
        # Medium-term period (t-20 to t)
        medium_close = close.iloc[i-20:i+1].values
        medium_volume = volume.iloc[i-20:i+1].values
        
        # Time indices for regression
        recent_time = np.arange(len(recent_close))
        medium_time = np.arange(len(medium_close))
        
        # Calculate price slopes
        recent_price_slope = linregress(recent_time, recent_close).slope
        medium_price_slope = linregress(medium_time, medium_close).slope
        
        # Calculate volume slopes
        recent_volume_slope = linregress(recent_time, recent_volume).slope
        medium_volume_slope = linregress(medium_time, medium_volume).slope
        
        # Avoid division by zero
        if abs(recent_volume_slope) < 1e-10:
            recent_volume_slope = 1e-10 * np.sign(recent_volume_slope) if recent_volume_slope != 0 else 1e-10
        
        if abs(medium_volume_slope) < 1e-10:
            medium_volume_slope = 1e-10 * np.sign(medium_volume_slope) if medium_volume_slope != 0 else 1e-10
        
        # Calculate divergence ratios
        recent_divergence = recent_price_slope / recent_volume_slope
        medium_divergence = medium_price_slope / medium_volume_slope
        
        # Combine both time horizons with equal weight
        factor_value = 0.5 * recent_divergence + 0.5 * medium_divergence
        factor_values.loc[current_date] = factor_value
    
    return factor_values
