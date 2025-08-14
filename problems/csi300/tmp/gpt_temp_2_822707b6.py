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
    df['range'] = df['high'] - df['low']
    df['volatility'] = df['range'].rolling(window=20).std()
    
    # Adjust Window Size Based on Volatility
    def adjust_window_size(volatility, high_threshold, low_threshold):
        if volatility > high_threshold:
            return 5  # Decrease window size
        elif volatility < low_threshold:
            return 30  # Increase window size
        else:
            return 20  # Default window size
    
    # Calculate Adaptive Window
    df['window_size'] = df['volatility'].apply(lambda x: adjust_window_size(x, df['volatility'].quantile(0.75), df['volatility'].quantile(0.25)))
    
    # Rolling Statistics with Adaptive Window
    df['rolling_mean'] = df.groupby('window_size')['volume_weighted_return'].transform(lambda x: x.rolling(min_periods=1, window=int(x.name)).mean())
    df['rolling_std'] = df.groupby('window_size')['volume_weighted_return'].transform(lambda x: x.rolling(min_periods=1, window=int(x.name)).std())
    
    # Final Alpha Factor
    df['alpha_factor'] = df['rolling_mean'] / df['rolling_std']
    
    return df['alpha_factor']
