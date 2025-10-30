import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Calculate Decaying Price Momentum
    # Compute Short-Term Price Return (5-day)
    short_return = (df['close'] / df['close'].shift(5) - 1)
    
    # Compute Medium-Term Price Return (20-day)
    medium_return = (df['close'] / df['close'].shift(20) - 1)
    
    # Apply Exponential Decay Weighting (higher weight on short-term)
    short_weight = 0.7
    medium_weight = 0.3
    decaying_momentum = short_weight * short_return + medium_weight * medium_return
    
    # Calculate Volume Trend Confirmation
    def calculate_volume_slope(volume_series, window):
        slopes = []
        for i in range(len(volume_series)):
            if i < window - 1:
                slopes.append(np.nan)
            else:
                window_data = volume_series.iloc[i-window+1:i+1]
                if len(window_data) == window:
                    x = np.arange(window)
                    slope, _, _, _, _ = linregress(x, window_data.values)
                    slopes.append(slope)
                else:
                    slopes.append(np.nan)
        return pd.Series(slopes, index=volume_series.index)
    
    # Compute Short-Term Volume Trend (5-day slope)
    short_volume_slope = calculate_volume_slope(df['volume'], 5)
    
    # Compute Medium-Term Volume Trend (20-day slope)
    medium_volume_slope = calculate_volume_slope(df['volume'], 20)
    
    # Generate Volume Confirmation Signal
    volume_confirmation = np.where(
        (short_volume_slope * medium_volume_slope) > 0,  # Same direction
        1,  # Convergence
        -1  # Divergence
    )
    volume_confirmation = pd.Series(volume_confirmation, index=df.index)
    
    # Combine Momentum and Volume Signals
    combined_signal = decaying_momentum * volume_confirmation
    
    # Apply Volume-Based Scaling
    # Use current volume as scaling factor, maintaining directional integrity
    volume_scaling = np.log1p(df['volume'])  # Log transform to reduce extreme values
    final_factor = combined_signal * volume_scaling
    
    return final_factor
