import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = df['open'].shift(-1) - df['close']
    
    # Volume Weighting
    df['volume_weighted_return'] = df['close_to_open_return'] * df['volume']
    
    # Determine Volatility
    df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
    fixed_window = 20
    df['volatility'] = df['hlc3'].rolling(window=fixed_window).std()
    
    # Adjust Window Size Based on Volatility
    rolling_window = (df['volatility'] > df['volatility'].median()).astype(int) * 10 + 30
    
    # Rolling Statistics with Adaptive Window
    def rolling_stats(series, window):
        return series.rolling(window=window, min_periods=1).agg(['mean', 'std'])
    
    df['rolling_mean'] = rolling_stats(df['volume_weighted_return'], rolling_window)['mean']
    df['rolling_std'] = rolling_stats(df['volume_weighted_return'], rolling_window)['std']
    
    # Final Alpha Factor
    df['alpha_factor'] = df['rolling_mean'] / df['rolling_std']
    
    return df['alpha_factor'].dropna()
