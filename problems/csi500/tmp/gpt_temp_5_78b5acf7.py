import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Compute Price Momentum
    # Short-Term Momentum (5-day)
    short_term_momentum = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    
    # Medium-Term Momentum (20-day)
    medium_term_momentum = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)
    
    # Compute Volume Momentum
    # Volume Ratio (current vs 5 days ago)
    volume_ratio = df['volume'] / df['volume'].shift(5)
    
    # Volume Trend (linear slope of last 5 days)
    def calc_volume_slope(volume_series):
        if len(volume_series) < 5 or volume_series.isna().any():
            return np.nan
        x = np.arange(5)
        slope, _, _, _, _ = linregress(x, volume_series.values)
        return slope
    
    volume_trend = df['volume'].rolling(window=5, min_periods=5).apply(
        calc_volume_slope, raw=False
    )
    
    # Normalize volume trend by average volume to make it comparable
    avg_volume = df['volume'].rolling(window=5, min_periods=5).mean()
    normalized_volume_trend = volume_trend / avg_volume
    
    # Combine Volume Signals
    volume_momentum = volume_ratio * normalized_volume_trend
    
    # Combine Signals
    # Check momentum alignment
    price_momentum_alignment = np.sign(short_term_momentum) * np.sign(medium_term_momentum)
    volume_price_alignment = np.sign(volume_momentum) * np.sign(short_term_momentum)
    
    # Generate Alpha Factor
    # Use short-term momentum as base, confirmed by volume and medium-term trend
    base_factor = short_term_momentum * volume_momentum
    
    # Apply sign correction based on alignment
    # Positive when all signals align, negative when they diverge
    alignment_multiplier = price_momentum_alignment * volume_price_alignment
    
    alpha_factor = base_factor * alignment_multiplier
    
    return alpha_factor
