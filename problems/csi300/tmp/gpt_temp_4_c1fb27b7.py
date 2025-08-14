import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics(df):
    # Compute Close-to-Open Return
    df['close_to_open_return'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Weight by Volume
    df['volume_weighted_return'] = df['close_to_open_return'] * df['volume']
    
    # Calculate Recent Volatility
    df['volatility'] = df['close'].rolling(window=20).std()
    
    # Define a function to adapt the window size based on recent volatility
    def adaptive_window_size(volatility, base_window=20):
        return int(base_window * (1 + 0.5 * (volatility / volatility.median() - 1)))
    
    # Calculate Short-Term Moving Average of Volume-Weighted Close-to-Open Return
    df['short_term_moving_avg'] = df.apply(lambda row: df['volume_weighted_return'].rolling(
        window=adaptive_window_size(row['volatility'], base_window=5)).mean(), axis=1)
    
    # Calculate Long-Term Moving Average of Volume-Weighted Close-to-Open Return
    df['long_term_moving_avg'] = df.apply(lambda row: df['volume_weighted_return'].rolling(
        window=adaptive_window_size(row['volatility'], base_window=20)).mean(), axis=1)
    
    # Final Alpha Factor
    df['alpha_factor'] = df['short_term_moving_avg'] - df['long_term_moving_avg']
    
    return df['alpha_factor'].dropna()
