import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price Momentum Component
    # Short-Term Momentum (5-day rate of change)
    short_term_momentum = (df['close'] / df['close'].shift(5) - 1)
    
    # Medium-Term Momentum (20-day rate of change)
    medium_term_momentum = (df['close'] / df['close'].shift(20) - 1)
    
    # Momentum Divergence Signal
    # Compare short-term vs medium-term momentum and calculate acceleration
    momentum_divergence = short_term_momentum - medium_term_momentum
    
    # Volume Confirmation Filter
    # Volume Trend Component - calculate volume trend slope using linear regression
    def volume_trend_slope(volume_series):
        x = np.arange(len(volume_series))
        slope = np.polyfit(x, volume_series, 1)[0]
        return slope
    
    # Calculate 20-day rolling volume trend slope
    volume_trend = df['volume'].rolling(window=20).apply(
        volume_trend_slope, raw=False
    )
    
    # Signal Enhancement
    # Multiply momentum divergence by volume trend
    raw_factor = momentum_divergence * volume_trend
    
    # Scale by absolute momentum strength
    momentum_strength = np.abs(short_term_momentum) + np.abs(medium_term_momentum)
    momentum_strength = momentum_strength.replace(0, np.nan)  # Avoid division by zero
    
    # Final factor
    factor = raw_factor / momentum_strength
    
    return factor
