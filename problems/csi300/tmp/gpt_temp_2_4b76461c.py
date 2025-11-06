import pandas as pd
import numpy as np

def heuristics_v2(df):
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    
    price_efficiency = (close - low.rolling(5).min()) / (high.rolling(5).max() - low.rolling(5).min())
    volume_acceleration = volume / volume.rolling(10).mean() - 1
    trend_persistence = np.sign(close.diff(3)) * close.pct_change(3).abs()
    
    directional_flow = price_efficiency * volume_acceleration * trend_persistence
    smoothed_flow = directional_flow.rolling(8).apply(lambda x: np.sum(x[x > 0]) - np.sum(x[x < 0]) if len(x) == 8 else np.nan)
    
    liquidity_gap = volume.rolling(15).std() / volume.rolling(15).mean()
    momentum_quality = trend_persistence.rolling(6).std()
    
    heuristics_matrix = smoothed_flow * liquidity_gap / (momentum_quality + 1e-8)
    
    return heuristics_matrix
