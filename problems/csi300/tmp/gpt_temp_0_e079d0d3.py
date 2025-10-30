import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Momentum Skewness Component
    # Calculate Short-Term Momentum (5-day price change)
    short_momentum = df['close'].pct_change(periods=5)
    
    # Calculate Medium-Term Momentum (20-day price change)
    medium_momentum = df['close'].pct_change(periods=20)
    
    # Calculate Momentum Skewness (ratio with direction)
    momentum_skewness = np.sign(short_momentum) * (short_momentum / (medium_momentum + 1e-8))
    
    # Volume Divergence Component
    # Calculate Volume Trend (20-day volume slope)
    def volume_slope(volume_series):
        if len(volume_series) < 20:
            return np.nan
        x = np.arange(len(volume_series))
        slope = linregress(x, volume_series.values).slope
        return slope
    
    volume_trend = df['volume'].rolling(window=20).apply(volume_slope, raw=False)
    
    # Calculate Price-Volume Divergence
    # Use 5-day momentum direction for comparison with volume trend
    momentum_direction = np.sign(short_momentum.rolling(window=5).mean())
    volume_direction = np.sign(volume_trend)
    
    # Generate Divergence Signal
    divergence_signal = momentum_direction * volume_direction * -1  # Negative when trends diverge
    
    # Quantify divergence strength using absolute momentum and volume trend
    divergence_strength = np.abs(short_momentum) * np.abs(volume_trend)
    volume_divergence = divergence_signal * divergence_strength
    
    # Combine Components
    # Multiply Momentum Skewness by Volume Divergence
    alpha_factor = momentum_skewness * volume_divergence
    
    # Apply scaling factor for normalization
    alpha_factor = alpha_factor / (alpha_factor.rolling(window=60).std() + 1e-8)
    
    return alpha_factor
