import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = df['open'].shift(-1) - df['close']
    
    # Volume Weighting
    df['volume_weighted_return'] = df['volume'] * df['close_to_open_return']
    
    # Determine Volatility using High, Low, and Close prices
    df['volatility'] = df[['high', 'low', 'close']].std(axis=1)
    volatility_rolling = df['volatility'].rolling(window=20).std()
    
    # Adjust Window Size Based on Volatility
    def adjust_window_size(volatility, low_vol_threshold, high_vol_threshold, low_vol_window, high_vol_window, mid_vol_window):
        if volatility < low_vol_threshold:
            return low_vol_window
        elif volatility > high_vol_threshold:
            return high_vol_window
        else:
            return mid_vol_window
    
    window_size = [adjust_window_size(v, 0.005, 0.015, 60, 10, 30) for v in volatility_rolling]
    df['window_size'] = window_size
    
    # Calculate Rolling Statistics
    rolling_mean = df['volume_weighted_return'].rolling(window=df['window_size'], min_periods=1).mean()
    rolling_std = df['volume_weighted_return'].rolling(window=df['window_size'], min_periods=1).std()
    
    # Final Alpha Factor: Z-Score of the Volume Weighted Close-to-Open Return
    df['alpha_factor'] = (df['volume_weighted_return'] - rolling_mean) / rolling_std
    
    return df['alpha_factor']
