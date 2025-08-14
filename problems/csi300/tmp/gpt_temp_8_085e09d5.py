import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = df['open'].shift(-1) - df['close']
    
    # Volume Weighting
    df['volume_weighted_return'] = df['close_to_open_return'] * df['volume']
    
    # Calculate True Range
    df['true_range'] = df[['high', 'low', 'close']].apply(
        lambda x: max(x['high'] - x['low'], abs(x['high'] - x['close'].shift(1)), abs(x['low'] - x['close'].shift(1))), axis=1)
    
    # Calculate Average True Range (20-day rolling)
    df['atr_20'] = df['true_range'].rolling(window=20).mean()
    
    # Calculate Standard Deviation of Close Prices (20-day rolling)
    df['std_close_20'] = df['close'].rolling(window=20).std()
    
    # Determine Adaptive Window Size
    def adaptive_window(atr, std_close):
        if atr > 1.5 * std_close:
            return 10
        elif atr < 0.5 * std_close:
            return 30
        else:
            return 20
    
    df['adaptive_window'] = df.apply(lambda x: adaptive_window(x['atr_20'], x['std_close_20']), axis=1)
    
    # Calculate Rolling Mean and Standard Deviation of Volume Weighted Close-to-Open Return
    df['rolling_mean'] = df.groupby('date')['volume_weighted_return'].transform(lambda x: x.rolling(df['adaptive_window']).mean())
    df['rolling_std'] = df.groupby('date')['volume_weighted_return'].transform(lambda x: x.rolling(df['adaptive_window']).std())
    
    # Normalize Volume Weighted Close-to-Open Return
    df['normalized_volume_weighted_return'] = (df['volume_weighted_return'] - df['rolling_mean']) / df['rolling_std']
    
    # Output as Alpha Factor
    alpha_factor = df['normalized_volume_weighted_return']
    
    return alpha_factor
