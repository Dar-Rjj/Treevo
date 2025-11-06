import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    price_momentum = close.pct_change(5)
    momentum_acceleration = price_momentum - price_momentum.rolling(window=3).mean()
    
    volume_trend = volume.rolling(window=8).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    volume_consistency = volume_trend / (volume.rolling(window=8).std() + 1e-8)
    
    volatility_normalization = (high - low).rolling(window=10).mean()
    
    heuristics_matrix = momentum_acceleration * volume_consistency / (volatility_normalization + 1e-8)
    
    return heuristics_matrix
