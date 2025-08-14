import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics(df):
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = df['open'].shift(-1) - df['close']
    
    # Volume Weighting
    df['volume_weighted_return'] = df['close_to_open_return'] * df['volume']
    
    # Determine Volatility (using High, Low, and Close prices)
    df['true_range'] = df[['high', 'low', 'close']].apply(lambda x: max(x) - min(x), axis=1)
    initial_window = 20
    df['volatility'] = df['true_range'].rolling(window=initial_window).std()
    
    # Adjust Window Size based on Volatility
    def adjust_window(vol):
        if vol > df['volatility'].mean():
            return int(initial_window * 0.8)  # Decrease window size
        else:
            return int(initial_window * 1.2)  # Increase window size
    
    df['window_size'] = df['volatility'].apply(adjust_window)
    
    # Calculate Rolling Statistics with Adaptive Window
    rolling_mean = df['volume_weighted_return'].rolling(window=df['window_size']).mean()
    rolling_std = df['volume_weighted_return'].rolling(window=df['window_size']).std()
    
    # Heuristic Factor
    df['heuristic_factor'] = (rolling_mean / rolling_std).fillna(0)
    
    return df['heuristic_factor']
