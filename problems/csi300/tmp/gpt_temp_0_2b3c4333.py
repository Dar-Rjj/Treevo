import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = df['open'].shift(-1) - df['close']
    
    # Volume Weighting
    df['volume_weighted_return'] = df['close_to_open_return'] * df['volume']
    
    # Adaptive Volatility Calculation
    df['daily_volatility'] = df['high'] - df['low']
    volatility_mean = df['daily_volatility'].rolling(window=20).mean()
    volatility_std = df['daily_volatility'].rolling(window=20).std()
    df['volatility_zscore'] = (df['daily_volatility'] - volatility_mean) / volatility_std
    
    # Adjust for Volatility
    df['adjusted_volatility'] = np.where(df['volatility_zscore'] > 0, 1.5, 0.5)
    
    # Rolling Statistics with Adaptive Volatility Window
    adaptive_window = (df['volatility_zscore'] * df['adjusted_volatility']).astype(int)
    df['rolling_mean'] = df['volume_weighted_return'].rolling(window=adaptive_window, min_periods=1).mean()
    df['rolling_std'] = df['volume_weighted_return'].rolling(window=adaptive_window, min_periods=1).std()
    
    # Final Alpha Factor
    df['alpha_factor'] = (df['volume_weighted_return'] - df['rolling_mean']) / df['rolling_std']
    
    return df['alpha_factor']
