import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    price_trend = close.rolling(10).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    volume_weighted_vol = (high - low).rolling(15).std() * volume.rolling(15).mean()
    
    trend_persistence = price_trend.rolling(8).std()
    volatility_compression = 1 / volume_weighted_vol.rolling(8).mean()
    
    liquidity_constrained_momentum = trend_persistence * volatility_compression
    
    heuristics_matrix = liquidity_constrained_momentum
    
    return heuristics_matrix
