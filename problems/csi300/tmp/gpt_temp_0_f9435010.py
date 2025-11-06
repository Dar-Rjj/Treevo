import pandas as pd
import numpy as np

def heuristics_v2(df):
    high, low, close, volume = df['high'], df['low'], df['close'], df['volume']
    
    volatility_persistence = (high.rolling(5).std() / low.rolling(5).std()) * (close / close.rolling(10).mean())
    volume_trend = volume.rolling(10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    momentum_stability = close.pct_change(3).rolling(8).std()
    
    heuristics_matrix = volatility_persistence * volume_trend / (momentum_stability + 1e-12)
    heuristics_matrix = heuristics_matrix.replace([np.inf, -np.inf], np.nan)
    
    return heuristics_matrix
