import pandas as pd
import numpy as np

def heuristics_v2(df):
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    open_ = df['open']
    
    momentum_acceleration = (close - close.shift(5)) / close.shift(5) - (close.shift(5) - close.shift(10)) / close.shift(10)
    volume_persistence = volume.rolling(5).apply(lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) == 5 and not np.isnan(x).any() and np.std(x) > 0 else 0)
    range_compression = (high.rolling(3).max() - low.rolling(3).min()) / (high.rolling(10).max() - low.rolling(10).min())
    
    heuristics_matrix = momentum_acceleration * volume_persistence / (range_compression + 1e-8)
    heuristics_matrix = heuristics_matrix.replace([float('inf'), -float('inf')], float('nan'))
    
    return heuristics_matrix
