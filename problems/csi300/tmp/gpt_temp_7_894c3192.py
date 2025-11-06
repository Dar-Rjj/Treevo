import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Short-term price efficiency (fractal efficiency)
    price_range_5 = (high.rolling(5).max() - low.rolling(5).min())
    net_move_5 = (close - close.shift(5)).abs()
    efficiency_5 = net_move_5 / (price_range_5 + 1e-8)
    
    # Long-term trend stability (rolling trend consistency)
    trend_20 = close.rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    trend_stability = 1.0 / (trend_20.rolling(10).std() + 1e-8)
    
    # Volume-flow divergence (price-volume direction alignment)
    price_direction = np.sign(close - close.shift(1))
    volume_direction = np.sign(volume - volume.shift(1))
    flow_divergence = (price_direction * volume_direction).rolling(5).sum()
    
    # Regime-adaptive combination
    heuristics_matrix = efficiency_5 * trend_stability * flow_divergence
    
    return heuristics_matrix
