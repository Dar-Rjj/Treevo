import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    price_volatility = (high - low).rolling(window=8).std()
    volatility_adjusted_ma = close.rolling(window=5).mean() / (price_volatility + 1e-8)
    price_deviation = (close - volatility_adjusted_ma) / close
    
    volume_trend = volume.rolling(window=8).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    volume_consistency = volume_trend.rolling(window=5).std()
    
    heuristics_matrix = price_deviation * volume_consistency
    
    return heuristics_matrix
