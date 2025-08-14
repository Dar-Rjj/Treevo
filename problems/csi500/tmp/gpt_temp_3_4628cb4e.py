import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Daily Price Change
    df['daily_price_change'] = df['close'].diff()

    # Compute Volume Adjusted Return
    df['volume_adjusted_return'] = df['daily_price_change'] / df['volume']

    # Initialize EMA for Volume Adjusted Returns
    N = 14
    multiplier = 2 / (N + 1)
    df['ema_volume_adjusted_return'] = 0.0
    df.iloc[N, df.columns.get_loc('ema_volume_adjusted_return')] = df.iloc[N-1]['volume_adjusted_return']
    for i in range(N+1, len(df)):
        df.loc[df.index[i], 'ema_volume_adjusted_return'] = (df.loc[df.index[i-1], 'volume_adjusted_return'] * multiplier) + (df.loc[df.index[i-1], 'ema_volume_adjusted_return'] * (1 - multiplier))

    # Generate Alpha Factor Components: High-Low Range
    df['high_low_range'] = df['high'] - df['low']

    # Calculate Volume Weighted High-Low Range
    df['volume_weighted_high_low_range'] = df['high_low_range'] * df['volume']

    # Initialize EMA for Volume Weighted High-Low Range
    df['ema_volume_weighted_high_low_range'] = 0.0
    df.iloc[N, df.columns.get_loc('ema_volume_weighted_high_low_range')] = df.iloc[N-1]['volume_weighted_high_low_range']
    for i in range(N+1, len(df)):
        df.loc[df.index[i], 'ema_volume_weighted_high_low_range'] = (df.loc[df.index[i-1], 'volume_weighted_high_low_range'] * multiplier) + (df.loc[df.index[i-1], 'ema_volume_weighted_high_low_range'] * (1 - multiplier))

    # Generate Final Alpha Factor
    df['alpha_factor'] = df['ema_volume_adjusted_return'] - df['ema_volume_weighted_high_low_range']

    # Incorporate Recent Price Change Adjustment
    df['recent_price_change'] = df['close'].pct_change()
    df['adjusted_alpha_factor'] = df['alpha_factor'] + df['recent_price_change']

    return df['adjusted_alpha_factor']
