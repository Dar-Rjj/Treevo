import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    volume_acceleration = volume / volume.rolling(10).mean() - 1
    price_trend = close.rolling(5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    volume_price_correlation = volume_acceleration.rolling(8).corr(price_trend)
    
    relative_strength = (close - low.rolling(10).min()) / (high.rolling(10).max() - low.rolling(10).min() + 1e-8)
    
    heuristics_matrix = volume_price_correlation * relative_strength
    
    return heuristics_matrix
