import pandas as pd
import numpy as np
import pandas as pd

def heuristics_v2(df, lookback_period=20, m=10):
    # Calculate Close-to-Low Spread
    close_to_low_spread = (df['close'] - df['low']).clip(lower=0)
    
    # Sum Volume over Period
    sum_volume = df['volume'].rolling(window=lookback_period).sum()
    
    # Cumulative Spread Over Period
    cumulative_spread = close_to_low_spread.rolling(window=lookback_period).sum()
    
    # Divide by Accumulated Volume
    volume_adjusted_spread = cumulative_spread / sum_volume
    volume_adjusted_spread = volume_adjusted_spread.replace([pd.np.inf, -pd.np.inf], 0).fillna(0)
    
    # Calculate Daily Range
    daily_range = df['high'] - df['low']
    
    # Compare with Previous Day's Range
    momentum = (daily_range > daily_range.shift(1)).astype(int)
    
    # Enhanced Volume-Amount Ratio Trend
    volume_amount_ratio = df['volume'] / df['amount']
    recent_ratio_sum = volume_amount_ratio.rolling(window=5).sum()
    previous_ratio_sum = volume_amount_ratio.shift(5).rolling(window=5).sum()
    trend = (recent_ratio_sum > previous_ratio_sum).astype(int) * 2 - 1  # 1 for upward, -1 for downward, 0 for no trend
    
    # Combine High-Low Range Momentum and Enhanced Volume-Amount Ratio Trend
    combined_trend = momentum * trend
    
    # Adjusted Relative Strength and Trend
    relative_strength_ratio = (df['close'] / df['open']).shift(1)
    adjusted_relative_strength = relative_strength_ratio * volume_adjusted_spread
    smoothed_trend = adjusted_relative_strength.ewm(span=m).mean()
    
    # Final Alpha Factor
    alpha_factor = smoothed_trend * combined_trend
    return alpha_factor
