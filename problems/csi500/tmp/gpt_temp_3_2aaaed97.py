import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, N=14):
    # Calculate Daily Price Change
    df['price_change'] = df['close'].diff()
    
    # Compute Volume Adjusted Return
    df['volume_adjusted_return'] = df['price_change'] / df['volume']
    
    # Initialize EMA of Volume Adjusted Returns
    ema_var = df['volume_adjusted_return'].copy()
    multiplier = 2 / (N + 1)
    ema_var[0] = df.loc[df.index[0], 'volume_adjusted_return']
    
    for i in range(1, len(df)):
        ema_var[i] = (df.loc[df.index[i], 'volume_adjusted_return'] * multiplier) + (ema_var[i-1] * (1 - multiplier))
    
    # Calculate High-Low Range
    df['high_low_range'] = df['high'] - df['low']
    
    # Calculate Volume Weighted High-Low Range
    df['volume_weighted_high_low_range'] = df['high_low_range'] * df['volume']
    
    # Initialize EMA of Volume Weighted High-Low Range
    ema_vwhlr = df['volume_weighted_high_low_range'].copy()
    ema_vwhlr[0] = df.loc[df.index[0], 'volume_weighted_high_low_range']
    
    for i in range(1, len(df)):
        ema_vwhlr[i] = (df.loc[df.index[i], 'volume_weighted_high_low_range'] * multiplier) + (ema_vwhlr[i-1] * (1 - multiplier))
    
    # Generate Final Alpha Factor
    df['alpha_factor'] = ema_var - ema_vwhlr
    
    return df['alpha_factor']

# Example usage:
# alpha_series = heuristics_v2(df)
