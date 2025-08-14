import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate High-Low Range
    high_low_range = df['high'] - df['low']
    
    # Calculate High-Low Range Momentum
    high_low_range_momentum = high_low_range - high_low_range.shift(1)
    
    # Calculate Daily Returns
    daily_returns = df['close'] - df['close'].shift(1)
    
    # Calculate 20-day Weighted Moving Average of Returns
    weighted_avg_returns = daily_returns.rolling(window=20, min_periods=1).mean()
    
    # Calculate Volume Change
    volume_change = df['volume'] - df['volume'].shift(1)
    
    # Combine High-Low Range Momentum and Volume Change
    combined_factor = high_low_range_momentum * np.abs(volume_change)
    
    # Adjust for Sign
    combined_factor = np.where(volume_change > 0, combined_factor, -combined_factor)
    
    # Detect and Adjust for Volume Spikes
    volume_20_day_avg = df['volume'].rolling(window=20, min_periods=1).mean()
    volume_spike = df['volume'] > 1.5 * volume_20_day_avg
    adjusted_weighted_avg_returns = np.where(volume_spike, 0.7 * weighted_avg_returns, weighted_avg_returns)
    
    # Add Adjusted 20-day Weighted Moving Average to Combined High-Low Range Momentum
    final_alpha_factor = combined_factor + adjusted_weighted_avg_returns
    
    return final_alpha_factor
