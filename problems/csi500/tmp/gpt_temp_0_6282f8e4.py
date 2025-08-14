import pandas as pd
import pandas as pd

def heuristics_v2(df, N=14):
    # Calculate Daily Price Change
    df['price_change'] = df['close'].diff()
    
    # Compute Volume Adjusted Return
    df['volume_adjusted_return'] = df['price_change'] / df['volume']
    
    # Initialize EMA for Volume Adjusted Returns
    df['ema_var'] = 0.0
    multiplier = 2 / (N + 1)
    df.loc[df.index[0], 'ema_var'] = df.loc[df.index[0], 'volume_adjusted_return']
    
    # Calculate EMA of Volume Adjusted Returns
    for i in range(1, len(df)):
        df.loc[df.index[i], 'ema_var'] = (df.loc[df.index[i], 'volume_adjusted_return'] * multiplier) + (df.loc[df.index[i-1], 'ema_var'] * (1 - multiplier))
    
    # Calculate High-Low Range
    df['high_low_range'] = df['high'] - df['low']
    
    # Calculate Volume Weighted High-Low Range
    df['volume_weighted_high_low_range'] = df['high_low_range'] * df['volume']
    
    # Initialize EMA for Volume Weighted High-Low Range
    df['ema_vwhl'] = 0.0
    df.loc[df.index[0], 'ema_vwhl'] = df.loc[df.index[0], 'volume_weighted_high_low_range']
    
    # Calculate EMA of Volume Weighted High-Low Range
    for i in range(1, len(df)):
        df.loc[df.index[i], 'ema_vwhl'] = (df.loc[df.index[i], 'volume_weighted_high_low_range'] * multiplier) + (df.loc[df.index[i-1], 'ema_vwhl'] * (1 - multiplier))
    
    # Generate Final Alpha Factor
    df['alpha_factor'] = df['ema_var'] - df['ema_vwhl']
    
    return df['alpha_factor']
