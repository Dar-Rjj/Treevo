import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    """
    Price-Momentum and Volume Divergence Factor
    Combines short-term vs medium-term price momentum with volume momentum
    """
    close = df['close']
    volume = df['volume']
    
    # Initialize result series
    factor = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i < 20:  # Need at least 20 periods for medium-term calculation
            factor.iloc[i] = 0
            continue
            
        # Calculate Price Momentum
        # Short-term price trend (t-5 to t)
        short_price_data = close.iloc[i-5:i+1]
        if len(short_price_data) >= 2:
            short_price_slope = linregress(range(len(short_price_data)), short_price_data.values)[0]
        else:
            short_price_slope = 0
            
        # Medium-term price trend (t-20 to t)
        medium_price_data = close.iloc[i-20:i+1]
        if len(medium_price_data) >= 2:
            medium_price_slope = linregress(range(len(medium_price_data)), medium_price_data.values)[0]
        else:
            medium_price_slope = 0
            
        # Calculate Volume Momentum
        # Short-term volume trend (t-5 to t)
        short_volume_data = volume.iloc[i-5:i+1]
        if len(short_volume_data) >= 2:
            short_volume_slope = linregress(range(len(short_volume_data)), short_volume_data.values)[0]
        else:
            short_volume_slope = 0
            
        # Medium-term volume trend (t-20 to t)
        medium_volume_data = volume.iloc[i-20:i+1]
        if len(medium_volume_data) >= 2:
            medium_volume_slope = linregress(range(len(medium_volume_data)), medium_volume_data.values)[0]
        else:
            medium_volume_slope = 0
            
        # Calculate Price Momentum Ratio
        if abs(medium_price_slope) > 1e-10:  # Avoid division by zero
            price_momentum_ratio = short_price_slope / medium_price_slope
        else:
            price_momentum_ratio = 0
            
        # Calculate Volume Momentum Ratio
        if abs(medium_volume_slope) > 1e-10:  # Avoid division by zero
            volume_momentum_ratio = short_volume_slope / medium_volume_slope
        else:
            volume_momentum_ratio = 0
            
        # Compute Final Factor
        factor_value = price_momentum_ratio * volume_momentum_ratio
        factor.iloc[i] = np.sign(factor_value) if abs(factor_value) > 1e-10 else 0
    
    return factor
