import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Momentum-Decay Volume Divergence factor that combines price momentum, 
    volume trends, and time decay to predict future returns.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Recent Price Momentum - 5-day close price exponential-weighted slope
    close_series = data['close']
    
    # Calculate exponential weights for 5-day window
    alpha = 2 / (5 + 1)  # Smoothing factor
    weights = [(1 - alpha) ** i for i in range(4, -1, -1)]
    weights = np.array(weights) / sum(weights)
    
    # Calculate weighted momentum using rolling window
    def calc_ew_slope(window):
        if len(window) < 5:
            return np.nan
        x = np.arange(5)
        y = window.values
        return np.sum(weights * (y - np.average(y, weights=weights)) * 
                     (x - np.average(x, weights=weights))) / np.sum(weights * (x - np.average(x, weights=weights)) ** 2)
    
    momentum = close_series.rolling(window=5, min_periods=5).apply(calc_ew_slope, raw=False)
    
    # 2. Volume Trend - 10-day volume linear regression slope
    volume_series = data['volume']
    
    def calc_volume_slope(window):
        if len(window) < 10:
            return np.nan
        x = np.arange(10)
        y = window.values
        return np.cov(x, y)[0, 1] / np.var(x)
    
    volume_trend = volume_series.rolling(window=10, min_periods=10).apply(calc_volume_slope, raw=False)
    
    # 3. Momentum-Volume Alignment
    # Sign comparison and multiplier based on alignment
    momentum_sign = np.sign(momentum)
    volume_sign = np.sign(volume_trend)
    
    # Alignment multiplier: positive when signs match, negative when they diverge
    alignment_multiplier = momentum_sign * volume_sign
    
    # Strength adjustment based on magnitude
    momentum_strength = np.abs(momentum) / momentum.rolling(window=20, min_periods=1).std()
    volume_strength = np.abs(volume_trend) / volume_trend.rolling(window=20, min_periods=1).std()
    
    combined_strength = momentum_strength * volume_strength
    
    # 4. Time Decay Application
    # Exponential decay weighting with half-life of 10 days
    decay_factor = 0.5 ** (1/10)
    
    # Calculate decay-weighted divergence
    divergence = alignment_multiplier * combined_strength
    
    # Apply exponential decay using expanding window
    def apply_decay(series):
        if len(series) < 5:
            return np.nan
        weights = np.array([decay_factor ** i for i in range(len(series)-1, -1, -1)])
        weights = weights / np.sum(weights)
        return np.sum(series.values * weights)
    
    # Use expanding window with minimum 5 periods
    final_factor = divergence.expanding(min_periods=5).apply(apply_decay, raw=False)
    
    return final_factor
