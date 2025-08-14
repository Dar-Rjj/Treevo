import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily High-Low Difference
    daily_high_low_diff = df['high'] - df['low']
    
    # Compute Weighted Sum of High-Low Differences
    weights = (df.index.get_level_values('date') - df.index.levels[0][0]).days + 1
    weighted_sum_high_low = (daily_high_low_diff * weights).cumsum()
    
    # Calculate Volume-Adjusted Momentum
    window_size = 5
    long_window_size = 20
    volume_momentum = df['volume'].pct_change(window_size)
    recent_volume = df['volume'].rolling(window=window_size).mean()
    long_term_volume = df['volume'].rolling(window=long_window_size).mean()
    volume_ratio = recent_volume / long_term_volume
    volume_adjusted_momentum = volume_momentum * volume_ratio
    
    # Adjust Weighted Cumulative Moving Difference by Volume-Adjusted Momentum
    alpha_factor = weighted_sum_high_low * volume_adjusted_momentum
    
    return alpha_factor
