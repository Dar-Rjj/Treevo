import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Daily Price Change
    df['daily_price_change'] = df['close'].diff()
    
    # Compute Volume Adjusted Return
    df['volume_adjusted_return'] = df['daily_price_change'] / df['volume']
    
    # Initialize EMA of Volume Adjusted Returns
    n = 14  # Number of days for EMA
    multiplier = 2 / (n + 1)
    df['ema_volume_adjusted_return'] = 0.0
    df.loc[df.index[0], 'ema_volume_adjusted_return'] = df.loc[df.index[0], 'volume_adjusted_return']
    
    # Calculate EMA of Volume Adjusted Returns
    for i in range(1, len(df)):
        ema_t = (df.loc[df.index[i], 'volume_adjusted_return'] * multiplier) + (df.loc[df.index[i-1], 'ema_volume_adjusted_return'] * (1 - multiplier))
        df.loc[df.index[i], 'ema_volume_adjusted_return'] = ema_t
    
    # Generate Adaptive High-Low Range
    df['high_low_range'] = df['high'] - df['low']
    df['volume_weighted_high_low_range'] = df['high_low_range'] * df['volume']
    
    # Initialize EMA of Volume Weighted High-Low Range
    df['ema_volume_weighted_high_low_range'] = 0.0
    df.loc[df.index[0], 'ema_volume_weighted_high_low_range'] = df.loc[df.index[0], 'volume_weighted_high_low_range']
    
    # Calculate EMA of Volume Weighted High-Low Range
    for i in range(1, len(df)):
        ema_t = (df.loc[df.index[i], 'volume_weighted_high_low_range'] * multiplier) + (df.loc[df.index[i-1], 'ema_volume_weighted_high_low_range'] * (1 - multiplier))
        df.loc[df.index[i], 'ema_volume_weighted_high_low_range'] = ema_t
    
    # Generate Final Alpha Factor
    df['alpha_factor'] = df['ema_volume_adjusted_return'] - df['ema_volume_weighted_high_low_range']
    
    return df['alpha_factor']
