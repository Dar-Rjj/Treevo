import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Momentum-Decay Volume Divergence factor
    Combines price momentum with volume trend divergence and applies time decay
    """
    # Calculate Recent Price Momentum (5-day exponential-weighted slope)
    close_prices = df['close']
    
    # Create exponential weights for 5-day window
    alpha = 2 / (5 + 1)
    weights = [(1 - alpha) ** i for i in range(4, -1, -1)]
    weights = np.array(weights) / sum(weights)
    
    def exp_weighted_slope(window):
        if len(window) < 5:
            return np.nan
        x = np.arange(5)
        y = window.values
        return np.sum(weights * (x - np.average(x, weights=weights)) * (y - np.average(y, weights=weights))) / np.sum(weights * (x - np.average(x, weights=weights)) ** 2)
    
    momentum = close_prices.rolling(window=5, min_periods=5).apply(exp_weighted_slope, raw=False)
    
    # Compute Volume Trend (10-day linear regression slope)
    volume = df['volume']
    
    def linear_slope(window):
        if len(window) < 10:
            return np.nan
        x = np.arange(len(window))
        slope, _, _, _, _ = stats.linregress(x, window.values)
        return slope
    
    volume_trend = volume.rolling(window=10, min_periods=10).apply(linear_slope, raw=False)
    
    # Calculate Momentum-Volume Divergence
    momentum_sign = np.sign(momentum)
    volume_trend_sign = np.sign(volume_trend)
    
    # Alignment multiplier: +1 when signs align, -1 when they diverge
    alignment_multiplier = momentum_sign * volume_trend_sign
    
    # Raw divergence (absolute momentum adjusted by alignment)
    divergence = momentum.abs() * alignment_multiplier
    
    # Apply Time Decay (exponential decay weighting)
    decay_factor = 0.9  # 10% decay per day
    decay_weights = np.array([decay_factor ** i for i in range(4, -1, -1)])
    decay_weights = decay_weights / decay_weights.sum()
    
    def apply_decay(window):
        if len(window) < 5:
            return np.nan
        return np.sum(window.values * decay_weights)
    
    # Apply decay to the divergence signal
    final_factor = divergence.rolling(window=5, min_periods=5).apply(apply_decay, raw=False)
    
    return final_factor
