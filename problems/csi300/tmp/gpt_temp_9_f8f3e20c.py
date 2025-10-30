import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(data):
    """
    Price-Volume Convergence Divergence Factor
    Combines price and volume momentum trends with position-based weighting
    """
    
    close = data['close']
    volume = data['volume']
    high = data['high']
    low = data['low']
    
    # Initialize result series
    factor_values = pd.Series(index=close.index, dtype=float)
    
    for i in range(8, len(close)):
        current_date = close.index[i]
        
        # Recent Price Momentum Component
        # Short-term price trend (t-3 to t)
        short_price_window = close.iloc[i-3:i+1]
        if len(short_price_window) >= 2:
            short_price_slope = linregress(range(len(short_price_window)), short_price_window.values)[0]
        else:
            short_price_slope = 0
            
        # Medium-term price trend (t-8 to t-4)
        medium_price_window = close.iloc[i-8:i-3]
        if len(medium_price_window) >= 2:
            medium_price_slope = linregress(range(len(medium_price_window)), medium_price_window.values)[0]
        else:
            medium_price_slope = 0
        
        # Recent Volume Momentum Component
        # Short-term volume trend (t-3 to t)
        short_volume_window = volume.iloc[i-3:i+1]
        if len(short_volume_window) >= 2:
            short_volume_slope = linregress(range(len(short_volume_window)), short_volume_window.values)[0]
        else:
            short_volume_slope = 0
            
        # Medium-term volume trend (t-8 to t-4)
        medium_volume_window = volume.iloc[i-8:i-3]
        if len(medium_volume_window) >= 2:
            medium_volume_slope = linregress(range(len(medium_volume_window)), medium_volume_window.values)[0]
        else:
            medium_volume_slope = 0
        
        # Convergence-Divergence Signal Generation
        price_acceleration = short_price_slope - medium_price_slope
        volume_acceleration = short_volume_slope - medium_volume_slope
        
        # Compute Convergence Score
        convergence_score = price_acceleration * volume_acceleration
        
        # Apply Directional Weighting
        # Calculate price position within recent range (t-5 to t)
        recent_high = high.iloc[i-5:i+1].max()
        recent_low = low.iloc[i-5:i+1].min()
        current_close = close.iloc[i]
        
        if recent_high != recent_low:
            position_percentile = (current_close - recent_low) / (recent_high - recent_low)
            # Scale to range [-1, 1] where 0.5 is neutral
            position_weight = 2 * (position_percentile - 0.5)
        else:
            position_weight = 0
        
        # Final Factor
        final_factor = convergence_score * position_weight
        factor_values.iloc[i] = final_factor
    
    return factor_values
