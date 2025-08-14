import pandas as pd
import pandas as pd

def heuristics_v2(df, short_period=5, long_period=20, decay_factor=0.9, recent_spreads_days=10):
    # Calculate Daily High-Low Range
    df['high_low_range'] = df['high'] - df['low']
    
    # Compute Volume-Weighted High-Low Range Over Short Period
    df['vol_weighted_high_low_range_short'] = (df['high_low_range'] * df['volume']).rolling(window=short_period).sum()
    
    # Compute Volume-Weighted High-Low Range Over Long Period
    df['vol_weighted_high_low_range_long'] = (df['high_low_range'] * df['volume']).rolling(window=long_period).sum()
    
    # Compute the High-Low Range Momentum
    df['high_low_range_momentum'] = df['vol_weighted_high_low_range_short'] / df['vol_weighted_high_low_range_long']
    
    # Compute Weighted Sum of Recent Spreads
    spreads_momentum = []
    for i in range(len(df)):
        weighted_sum = 0
        cumulative_decay = 1
        for j in range(1, recent_spreads_days + 1):
            if i - j >= 0:
                cumulative_decay *= decay_factor
                weighted_sum += df.iloc[i - j]['high_low_range'] * cumulative_decay
        spreads_momentum.append(weighted_sum)
    
    df['spreads_momentum'] = spreads_momentum
    
    # Combine with Volume and Close-Open Return for Weighting
    df['close_open_return'] = (df['close'] - df['open']) / df['open']
    df['alpha_factor'] = df['high_low_range_momentum'] * df['spreads_momentum'] * df['volume'] * df['close_open_return']
    
    return df['alpha_factor']
