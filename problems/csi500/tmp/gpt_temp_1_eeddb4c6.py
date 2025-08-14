import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate High-Low Range
    high_low_range = df['high'] - df['low']
    
    # Calculate Daily Returns
    daily_returns = df['close'].pct_change()
    
    # Calculate Close Position in Range
    close_position_in_range = (df['close'] - df['low']) / (df['high'] - df['low'])
    
    # Calculate Volume-Weighted Position
    volume_weighted_position = close_position_in_range * df['volume']
    
    # Apply Exponential Smoothing to Volume-Weighted Position
    exponentially_smoothed_position = volume_weighted_position.ewm(alpha=0.2).mean()
    
    # Calculate Moving Average of Exponentially Smoothed Position
    ma_exponentially_smoothed_position = exponentially_smoothed_position.rolling(window=20).mean()
    
    # Combine High-Low Range Momentum and Volume-Weighted Position
    high_low_range_momentum = high_low_range - high_low_range.shift(1)
    combined_high_low_range_momentum = high_low_range_momentum * volume_weighted_position
    
    # Adjust for Sign
    combined_high_low_range_momentum = combined_high_low_range_momentum.apply(lambda x: x if volume_weighted_position > 0 else -x)
    
    # Adjust for Volume Spikes
    volume_20_day_ma = df['volume'].rolling(window=20).mean()
    volume_spike_days = df['volume'] > volume_20_day_ma
    adjusted_short_term_ma = ma_exponentially_smoothed_position.where(~volume_spike_days, ma_exponentially_smoothed_position * 0.5)
    
    # Introduce Trend Following Component
    long_term_ma_close = df['close'].rolling(window=100).mean()
    trend_bias = (df['close'] > long_term_ma_close).astype(int) * 1 - (df['close'] < long_term_ma_close).astype(int) * 1
    
    # Combine Adjusted Short-Term Moving Average, Combined High-Low Range Momentum, and Trend Following Component
    final_alpha_factor = adjusted_short_term_ma + combined_high_low_range_momentum + trend_bias
    
    return final_alpha_factor
