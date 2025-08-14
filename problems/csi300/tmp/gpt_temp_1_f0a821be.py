import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Close-to-Open Return
    df['Close_to_Open_Return'] = (df['open'].shift(-1) - df['close']) / df['close'].shift(1)
    
    # Volume Weighting
    df['Volume_Weighted_Return'] = df['Volume_Weighted_Return'] * df['volume']
    
    # Determine Volatility using High, Low, and Close prices
    df['HL_C_diff'] = (df['high'] + df['low'] + df['close']) / 3
    volatility_window = 30  # Fixed window for volatility calculation
    df['volatility'] = df['HL_C_diff'].rolling(window=volatility_window).std()
    
    # Adjust Window Size based on Volatility
    def adjust_window_size(vol):
        if vol > df['volatility'].mean():
            return int(volatility_window * 0.8)  # Decrease window size
        else:
            return int(volatility_window * 1.2)  # Increase window size
    
    df['window_size'] = df['volatility'].apply(adjust_window_size)
    
    # Calculate Rolling Statistics with Adaptive Window
    df['rolling_mean'] = df.groupby('window_size')['Volume_Weighted_Return'].transform(lambda x: x.rolling(window=x.name).mean())
    df['rolling_std'] = df.groupby('window_size')['Volume_Weighted_Return'].transform(lambda x: x.rolling(window=x.name).std())
    df['rolling_z_score'] = (df['Volume_Weighted_Return'] - df['rolling_mean']) / df['rolling_std']
    
    # Output the final alpha factor
    alpha_factor = df['rolling_z_score']
    
    return alpha_factor
