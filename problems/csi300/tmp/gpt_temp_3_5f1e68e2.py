import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    momentum_5 = close.pct_change(5)
    momentum_10 = close.pct_change(10)
    momentum_20 = close.pct_change(20)
    
    vol_adj_momentum = momentum_20 / (close.rolling(20).std() + 1e-8)
    
    price_acceleration = momentum_5 - momentum_10
    
    volume_trend = volume.rolling(5).mean() / volume.rolling(20).mean()
    
    high_low_range = (high.rolling(5).max() - low.rolling(5).min()) / close
    
    heuristics_matrix = (price_acceleration * vol_adj_momentum * 
                        volume_trend / (high_low_range + 1e-8))
    
    return heuristics_matrix
