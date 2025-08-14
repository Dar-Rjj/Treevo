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
    df['high_low_close_mean'] = (df['high'] + df['low'] + df['close']) / 3
    initial_window = 20  # Fixed initial window for volatility calculation
    df['volatility'] = df['high_low_close_mean'].rolling(window=initial_window).std()
    
    # Adjust Window Size based on Volatility
    df['window_size'] = np.where(df['volatility'] > df['volatility'].mean(), initial_window // 2, initial_window * 2)
    
    # Rolling Statistics with Adaptive Window
    def rolling_stats(s, window_sizes):
        mean = s.rolling(window=window_sizes, min_periods=1).mean()
        std = s.rolling(window=window_sizes, min_periods=1).std()
        return mean, std
    
    df['rolling_mean'], df['rolling_std'] = rolling_stats(df['volume_weighted_return'], df['window_size'])
    
    # Final Alpha Factor
    df['alpha_factor'] = df['rolling_mean'] / df['rolling_std']
    
    return df['alpha_factor']
