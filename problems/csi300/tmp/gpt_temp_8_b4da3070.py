import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(data):
    """
    Trend Acceleration with Volume Divergence factor
    Combines price trend acceleration with volume-price alignment
    """
    df = data.copy()
    
    # Initialize result series
    factor_values = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i < 20:  # Need at least 20 periods for medium-term trend
            factor_values.iloc[i] = 0
            continue
            
        current_data = df.iloc[:i+1]
        
        # Calculate Short-Term Trend (5 periods)
        if i >= 4:
            short_term_prices = current_data['close'].iloc[-5:]
            short_term_idx = np.arange(len(short_term_prices))
            short_slope, _, _, _, _ = linregress(short_term_idx, short_term_prices)
        else:
            short_slope = 0
        
        # Calculate Medium-Term Trend (20 periods)
        medium_term_prices = current_data['close'].iloc[-20:]
        medium_term_idx = np.arange(len(medium_term_prices))
        medium_slope, _, _, _, _ = linregress(medium_term_idx, medium_term_prices)
        
        # Calculate Trend Acceleration
        if abs(medium_slope) > 1e-10:  # Avoid division by zero
            acceleration = np.log(abs(short_slope / medium_slope))
        else:
            acceleration = 0
        
        # Calculate Volume Trend (5 periods)
        if i >= 4:
            short_term_volume = current_data['volume'].iloc[-5:]
            volume_idx = np.arange(len(short_term_volume))
            volume_slope, _, _, _, _ = linregress(volume_idx, short_term_volume)
        else:
            volume_slope = 0
        
        # Volume-Price Alignment
        volume_price_alignment = np.sign(volume_slope * short_slope)
        
        # Combine Acceleration with Volume-Price Alignment
        combined_factor = acceleration * volume_price_alignment
        
        # Apply hyperbolic tangent for bounded output
        factor_values.iloc[i] = np.tanh(combined_factor)
    
    return factor_values
