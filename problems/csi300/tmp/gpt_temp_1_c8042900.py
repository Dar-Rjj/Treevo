import pandas as pd
import numpy as np

def heuristics_v2(df):
    high = df['high']
    low = df['low']
    close = df['close']
    open_ = df['open']
    volume = df['volume']
    amount = df['amount']
    
    # Abnormal volume clustering (current volume vs recent extreme volumes)
    volume_quantile = volume.rolling(20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    volume_clustering = volume_quantile * np.sign(volume - volume.rolling(20).mean())
    
    # Price acceleration divergence (rate of change of momentum)
    price_momentum = close / close.shift(5) - 1
    acceleration = price_momentum - price_momentum.shift(3)
    acceleration_divergence = acceleration / acceleration.rolling(10).std()
    
    # Short-term trend exhaustion (resistance to recent highs/lows)
    resistance_level = high.rolling(10).max()
    support_level = low.rolling(10).min()
    trend_exhaustion = ((close - support_level) / (resistance_level - support_level) - 0.5) * 2
    
    # Combine factors with equal weighting
    heuristics_matrix = volume_clustering + acceleration_divergence + trend_exhaustion
    
    return heuristics_matrix
