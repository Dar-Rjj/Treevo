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
    df['high_low_close_std'] = df[['high', 'low', 'close']].std(axis=1)
    initial_window = 20
    df['volatility'] = df['high_low_close_std'].rolling(window=initial_window).std()
    
    # Adjust Window Size based on Volatility
    def adjust_window(vol):
        if vol > df['volatility'].quantile(0.75):
            return 10  # Decrease window size for high volatility
        elif vol < df['volatility'].quantile(0.25):
            return 30  # Increase window size for low volatility
        else:
            return initial_window  # Keep the initial window
    
    df['window_size'] = df['volatility'].apply(adjust_window)
    
    # Rolling Statistics with Adaptive Window
    def rolling_stats(group):
        window = int(group['window_size'].iloc[0])
        group['rolling_mean'] = group['volume_weighted_return'].rolling(window=window).mean()
        group['rolling_std'] = group['volume_weighted_return'].rolling(window=window).std()
        return group
    
    # Apply rolling statistics with adaptive window
    df = df.groupby('window_size').apply(rolling_stats)
    
    # Output the factor values (e.g., rolling mean of volume weighted close-to-open return)
    factor = df['rolling_mean'].dropna()
    return factor
