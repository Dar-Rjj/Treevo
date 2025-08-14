import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Calculate High-Low Range
    high_low_range = df['high'] - df['low']
    
    # Adjust for Opening Gap
    opening_gap = df['open'] - df['close'].shift(1)
    adjusted_range = high_low_range - opening_gap.abs()
    
    # Incorporate Volume Momentum
    volume_change = df['volume'] - df['volume'].shift(1)
    volume_weighted_range = adjusted_range * volume_change
    
    # Apply Time Series Momentum
    lookback_period = 10
    ma_volume_weighted_range = volume_weighted_range.rolling(window=lookback_period).mean()
    ts_momentum_diff = volume_weighted_range - ma_volume_weighted_range
    
    # Calculate Close-to-Open Return
    close_to_open_return = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Calculate Daily Volatility
    daily_volatility = (df['high'] - df['low']).abs()
    
    # Combine Close-to-Open Return with Volume
    combined_factor = (close_to_open_return * df['volume']) / daily_volatility
    
    # Volume Adjusted Momentum
    close_prices_last_5_days = df['close'].rolling(window=5).apply(lambda x: np.sum(np.diff(x)), raw=False)
    avg_volume_last_5_days = df['volume'].rolling(window=5).mean()
    volume_adjusted_momentum = close_prices_last_5_days * avg_volume_last_5_days
    
    # Momentum and Volume-Adjusted Close-to-Open Return
    momentum_and_volume_adjusted_return = combined_factor - volume_adjusted_momentum
    
    # Trend Analysis
    def calculate_slope(series, window):
        return series.rolling(window=window).apply(lambda x: linregress(np.arange(len(x)), x)[0], raw=False)
    
    five_day_trend = calculate_slope(df['close'], 5)
    twenty_day_trend = calculate_slope(df['close'], 20)
    combined_trends = five_day_trend - twenty_day_trend
    
    # Final Alpha Factor
    final_alpha_factor = ts_momentum_diff + volume_adjusted_momentum + momentum_and_volume_adjusted_return + combined_trends
    
    return final_alpha_factor
