import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Calculate Price Momentum
    # Short-term return (5-day)
    short_term_return = df['close'] / df['close'].shift(5) - 1
    
    # Medium-term return (20-day)
    medium_term_return = df['close'] / df['close'].shift(20) - 1
    
    # Combined price momentum (weighted average)
    price_momentum = 0.6 * short_term_return + 0.4 * medium_term_return
    
    # Calculate Volume Momentum
    # Volume change (5-day)
    volume_change = df['volume'] / df['volume'].shift(5) - 1
    
    # Volume trend (20-day linear slope)
    def calc_volume_slope(volume_series):
        if len(volume_series) < 20 or volume_series.isna().any():
            return np.nan
        x = np.arange(len(volume_series))
        slope, _, _, _, _ = linregress(x, volume_series)
        return slope
    
    volume_trend = df['volume'].rolling(window=20, min_periods=20).apply(
        calc_volume_slope, raw=False
    )
    
    # Combined volume momentum
    volume_momentum = 0.7 * volume_change + 0.3 * volume_trend
    
    # Combine Signals
    # Multiply price momentum by volume momentum
    combined_signal = price_momentum * volume_momentum
    
    # Apply sign function and scale by current volume
    factor = np.sign(combined_signal) * df['volume']
    
    return factor
