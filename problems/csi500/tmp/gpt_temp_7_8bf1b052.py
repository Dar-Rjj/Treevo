import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    price_gap = (close - close.shift(1)) / close.shift(1)
    volatility_scale = (high - low).rolling(10).std()
    scaled_gap_momentum = (price_gap / volatility_scale).rolling(5).sum()
    
    volume_trend = volume.rolling(8).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    gap_reversal = -price_gap.rolling(3).mean() * np.sign(volume_trend)
    
    heuristics_matrix = scaled_gap_momentum + gap_reversal
    
    return heuristics_matrix
