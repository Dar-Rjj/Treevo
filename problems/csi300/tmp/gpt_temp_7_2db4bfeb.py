import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    price_oscillation = (high - low).rolling(window=5).std() / close.rolling(window=5).mean()
    volume_trend = volume.rolling(window=8).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    volatility_clustering = close.pct_change().rolling(window=6).std()
    
    heuristics_matrix = price_oscillation * volume_trend / (volatility_clustering + 1e-8)
    return heuristics_matrix
