import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    price_position = (close - low.rolling(window=8).min()) / (high.rolling(window=8).max() - low.rolling(window=8).min() + 1e-8)
    mean_position = price_position.rolling(window=5).mean()
    
    volatility_persistence = (high - low).rolling(window=5).std() / (high - low).rolling(window=15).std()
    
    volume_trend = volume.rolling(window=5).apply(lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 else 0)
    
    heuristics_matrix = (price_position - mean_position) * volatility_persistence * volume_trend
    
    return heuristics_matrix
