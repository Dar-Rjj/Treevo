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
    n_days_ema = 14
    multiplier = 2 / (n_days_ema + 1)
    df['ema_volume_adjusted_return'] = 0.0
    df.iloc[0, df.columns.get_loc('ema_volume_adjusted_return')] = df.iloc[0]['volume_adjusted_return']
    for i in range(1, len(df)):
        df.iloc[i, df.columns.get_loc('ema_volume_adjusted_return')] = (df.iloc[i]['volume_adjusted_return'] * multiplier) + (df.iloc[i-1]['ema_volume_adjusted_return'] * (1 - multiplier))
    
    # Calculate High-Low Range
    df['high_low_range'] = df['high'] - df['low']
    
    # Calculate Volume Weighted High-Low Range
    df['volume_weighted_high_low_range'] = df['high_low_range'] * df['volume']
    
    # Initialize EMA of Volume Weighted High-Low Range
    df['ema_volume_weighted_high_low_range'] = 0.0
    df.iloc[0, df.columns.get_loc('ema_volume_weighted_high_low_range')] = df.iloc[0]['volume_weighted_high_low_range']
    for i in range(1, len(df)):
        df.iloc[i, df.columns.get_loc('ema_volume_weighted_high_low_range')] = (df.iloc[i]['volume_weighted_high_low_range'] * multiplier) + (df.iloc[i-1]['ema_volume_weighted_high_low_range'] * (1 - multiplier))
    
    # Integrate Trend and Volatility
    df['combined_ema'] = df['ema_volume_adjusted_return'] - df['ema_volume_weighted_high_low_range']
    
    # Generate Final Alpha Factor
    alpha_factor = df['combined_ema']
    
    return alpha_factor
