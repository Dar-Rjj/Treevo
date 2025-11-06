import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    volume = df['volume']
    
    price_velocity = close.pct_change(periods=5)
    acceleration = price_velocity - price_velocity.rolling(window=10).mean()
    volume_trend = volume.rolling(window=5).mean() / volume.rolling(window=20).mean()
    trend_strength = close.rolling(window=10).apply(lambda x: (x[-1] - x[0]) / (x.max() - x.min()))
    
    momentum_acceleration = acceleration * volume_trend * trend_strength
    heuristics_matrix = momentum_acceleration
    
    return heuristics_matrix
