import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = df['open'].shift(-1) - df['close']
    
    # Volume Weighting
    df['volume_weighted_return'] = df['close_to_open_return'] * df['volume']
    
    # Determine Volatility using High, Low, and Close prices
    df['volatility'] = df[['high', 'low', 'close']].rolling(window=20).std().mean(axis=1)
    
    # Adjust Window Size based on Volatility
    def adjust_window(vol):
        if vol > df['volatility'].quantile(0.75):  # High Volatility
            return 10
        elif vol < df['volatility'].quantile(0.25):  # Low Volatility
            return 30
        else:
            return 20  # Default window size
    
    df['window_size'] = df['volatility'].apply(adjust_window)
    
    # Rolling Statistics with Adaptive Window
    adaptive_mean = df['volume_weighted_return'].rolling(window=df['window_size']).mean()
    adaptive_std = df['volume_weighted_return'].rolling(window=df['window_size']).std()
    
    # Final Factor: Standardized Rolling Mean of Volume Weighted Close-to-Open Return
    df['factor'] = (adaptive_mean - adaptive_mean.rolling(window=20).mean()) / adaptive_mean.rolling(window=20).std()
    
    return df['factor']
