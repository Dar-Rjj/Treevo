import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, n=14):
    # Calculate Daily Price Change
    df['price_change'] = df['close'].diff()
    
    # Compute Volume Adjusted Return
    df['volume_adjusted_return'] = df['price_change'] / df['volume']
    
    # Initialize EMA of Volume Adjusted Returns
    ema_var = df['volume_adjusted_return'].iloc[0]
    df['ema_volume_adjusted_return'] = 0
    multiplier = 2 / (n + 1)
    
    for i in range(1, len(df)):
        ema_var = (df.loc[df.index[i], 'volume_adjusted_return'] * multiplier) + (ema_var * (1 - multiplier))
        df.loc[df.index[i], 'ema_volume_adjusted_return'] = ema_var
    
    # Calculate High-Low Range
    df['high_low_range'] = df['high'] - df['low']
    
    # Calculate Volume Weighted High-Low Range
    df['volume_weighted_high_low_range'] = df['high_low_range'] * df['volume']
    
    # Initialize EMA of Volume Weighted High-Low Range
    ema_vwhlr = df['volume_weighted_high_low_range'].iloc[0]
    df['ema_volume_weighted_high_low_range'] = 0
    
    for i in range(1, len(df)):
        ema_vwhlr = (df.loc[df.index[i], 'volume_weighted_high_low_range'] * multiplier) + (ema_vwhlr * (1 - multiplier))
        df.loc[df.index[i], 'ema_volume_weighted_high_low_range'] = ema_vwhlr
    
    # Generate Final Alpha Factor
    df['alpha_factor'] = df['ema_volume_adjusted_return'] - df['ema_volume_weighted_high_low_range']
    
    # Incorporate Recent Price Change Adjustment
    df['recent_price_change'] = df['close'] - df['close'].shift(1)
    df['final_alpha_factor'] = df['alpha_factor'] + df['recent_price_change']
    
    return df['final_alpha_factor']
