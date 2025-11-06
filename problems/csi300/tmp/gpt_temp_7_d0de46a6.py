import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    price_acceleration = close.pct_change(3) - close.pct_change(8)
    volatility = (high - low).rolling(window=10).mean()
    volatility_scaled_momentum = close.pct_change(12) * volatility
    
    momentum_divergence = price_acceleration - volatility_scaled_momentum
    
    volume_trend = volume.rolling(window=5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    volume_consistency = volume.rolling(window=10).std() / volume.rolling(window=10).mean()
    volume_weight = volume_trend / (volume_consistency + 1e-8)
    
    heuristics_matrix = momentum_divergence * volume_weight
    
    return heuristics_matrix
