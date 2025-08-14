import pandas as pd
import pandas as pd

def heuristics_v2(df, n=14):
    # Calculate Daily Price Change
    df['price_change'] = df['close'].diff()
    
    # Compute Volume Adjusted Return
    df['volume_adjusted_return'] = df['price_change'] / df['volume']
    
    # Initialize EMA of Volume Adjusted Returns
    multiplier = 2 / (n + 1)
    df['ema_volume_adjusted_return'] = 0.0
    df.loc[0, 'ema_volume_adjusted_return'] = df.loc[0, 'volume_adjusted_return']
    
    for i in range(1, len(df)):
        df.loc[i, 'ema_volume_adjusted_return'] = (df.loc[i, 'volume_adjusted_return'] * multiplier) + (df.loc[i-1, 'ema_volume_adjusted_return'] * (1 - multiplier))
    
    # Generate Alpha Factor
    # Calculate High-Low Range
    df['high_low_range'] = df['high'] - df['low']
    
    # Calculate Volume Weighted High-Low Range
    df['volume_weighted_high_low_range'] = df['high_low_range'] * df['volume']
    
    # Initialize EMA of Volume Weighted High-Low Range
    df['ema_volume_weighted_high_low_range'] = 0.0
    df.loc[0, 'ema_volume_weighted_high_low_range'] = df.loc[0, 'volume_weighted_high_low_range']
    
    for i in range(1, len(df)):
        df.loc[i, 'ema_volume_weighted_high_low_range'] = (df.loc[i, 'volume_weighted_high_low_range'] * multiplier) + (df.loc[i-1, 'ema_volume_weighted_high_low_range'] * (1 - multiplier))
    
    # Generate Final Alpha Factor
    df['alpha_factor'] = df['ema_volume_adjusted_return'] - df['ema_volume_weighted_high_low_range']
    
    return df['alpha_factor']
