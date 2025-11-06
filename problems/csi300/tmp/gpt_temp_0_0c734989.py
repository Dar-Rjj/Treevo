import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Short-term momentum acceleration (5-day vs 10-day momentum)
    mom_5 = close.pct_change(5)
    mom_10 = close.pct_change(10)
    accel_factor = mom_5 - mom_10
    
    # Volatility-adjusted long-term momentum (20-day)
    vol_20 = (high / low).rolling(20).std()
    mom_20 = close.pct_change(20)
    vol_adj_momentum = mom_20 / (vol_20 + 1e-8)
    
    # Volume confirmation signal
    volume_ma_ratio = volume / volume.rolling(10).mean()
    volume_trend = volume_ma_ratio.rolling(5).std()
    
    # Combine components with non-linear transformation
    raw_factor = (accel_factor * vol_adj_momentum * np.tanh(volume_trend))
    
    # Regime detection using rolling percentiles
    regime_signal = raw_factor.rolling(10).apply(lambda x: np.percentile(x, 70) - np.percentile(x, 30))
    
    heuristics_matrix = raw_factor * np.sign(regime_signal)
    return heuristics_matrix
