import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    short_term_reversal = -close.pct_change(5)
    
    volatility_clustering = (high - low).rolling(window=10).std()
    volume_acceleration = volume / volume.rolling(window=8).mean() - 1
    regime_signal = volatility_clustering * volume_acceleration
    
    volume_trend = volume.rolling(window=6).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    price_momentum = close.pct_change(3)
    trend_alignment = np.sign(price_momentum) * volume_trend
    
    heuristics_matrix = short_term_reversal + regime_signal + trend_alignment
    
    return heuristics_matrix
