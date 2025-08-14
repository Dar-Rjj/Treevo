import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Define lookback periods and EMA smoothing factors
    momentum_lookback = 30
    volume_ratio_lookback = 30
    momentum_ema_span = 10
    volume_ratio_ema_span = 10
    
    # Calculate Price Momentum
    df['price_diff'] = df['close'].diff()
    df['momentum'] = df['price_diff'].ewm(span=momentum_ema_span, adjust=False).mean()
    
    # Adjust by Volume-Weighted Change
    df['volume_ratio'] = (df['volume'] / df['volume'].shift(1)).fillna(1)
    df['smoothed_volume_ratio'] = df['volume_ratio'].ewm(span=volume_ratio_ema_span, adjust=False).mean()
    
    # Combine Momentum and Volume-Weighted Ratio
    df['combined_momentum'] = df['momentum'] * df['smoothed_volume_ratio']
    
    # Calculate High-Low Price Difference
    df['high_low_diff'] = df['high'] - df['low']
    
    # Compute Volume Influence Ratio
    df['upward_volume'] = np.where(df['close'] > df['open'], df['volume'], 0).rolling(window=volume_ratio_lookback).sum()
    df['downward_volume'] = np.where(df['close'] < df['open'], df['volume'], 0).rolling(window=volume_ratio_lookback).sum()
    df['volume_influence_ratio'] = df['upward_volume'] / df['downward_volume']
    
    # Integrate Combined Momentum, Volume-Weighted Ratio, and Directional Volume Impact
    df['alpha_factor'] = df['combined_momentum'] * df['volume_influence_ratio'] * df['high_low_diff']
    
    return df['alpha_factor']
