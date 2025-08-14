import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate High-Low Range
    high_low_range = df['high'] - df['low']
    
    # Calculate High-Low Range Momentum
    high_low_range_prev = high_low_range.shift(1)
    high_low_range_momentum = high_low_range - high_low_range_prev
    
    # Calculate Daily Returns
    daily_returns = df['close'].pct_change()
    
    # Calculate 10-day Moving Average of Returns
    short_term_ma_returns = daily_returns.rolling(window=10).mean()
    
    # Calculate Close Position in Range
    close_position_in_range = (df['close'] - df['low']) / high_low_range
    
    # Calculate Volume-Weighted Position
    volume_weighted_position = close_position_in_range * df['volume']
    
    # Apply Exponential Smoothing to Volume-Weighted Position
    alpha = 0.2
    exp_smoothed_position = volume_weighted_position.ewm(alpha=alpha, adjust=False).mean()
    
    # Calculate 20-day Moving Average of Exponentially Smoothed Position
    smoothed_ma = exp_smoothed_position.rolling(window=20).mean()
    
    # Combine High-Low Range Momentum and Volume-Weighted Position
    combined_factor = high_low_range_momentum * smoothed_ma
    combined_factor = np.where(smoothed_ma > 0, combined_factor, -combined_factor)
    
    # Adjust for Volume Spikes
    volume_ma_20 = df['volume'].rolling(window=20).mean()
    volume_spike_days = df['volume'] > 1.5 * volume_ma_20
    spike_adjustment_factor = 0.5
    adjusted_short_term_ma_returns = np.where(volume_spike_days, short_term_ma_returns * spike_adjustment_factor, short_term_ma_returns)
    
    # Combine Adjusted Short-Term Moving Average and Combined High-Low Range Momentum
    final_alpha_factor = combined_factor + adjusted_short_term_ma_returns
    
    return final_alpha_factor
