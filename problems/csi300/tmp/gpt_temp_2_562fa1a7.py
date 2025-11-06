import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Volatility regime detection using multiple timeframes
    vol_5 = close.pct_change().rolling(5).std()
    vol_20 = close.pct_change().rolling(20).std()
    vol_regime = vol_5 / vol_20
    
    # Trend stability measure using price channel efficiency
    high_10 = high.rolling(10).max()
    low_10 = low.rolling(10).min()
    channel_range = (high_10 - low_10) / close
    actual_range = (high - low) / close
    trend_efficiency = channel_range / (actual_range.rolling(10).mean() + 1e-8)
    
    # Volume clustering anomaly detection
    volume_z = (volume - volume.rolling(30).mean()) / (volume.rolling(30).std() + 1e-8)
    volume_clusters = volume_z.rolling(5).apply(lambda x: np.sum(np.abs(x) > 1.5))
    
    # Regime-dependent efficiency factor
    regime_factor = np.where(vol_regime > 1.2, trend_efficiency, -trend_efficiency)
    efficiency_gap = regime_factor * (1 + 0.1 * volume_clusters)
    
    heuristics_matrix = efficiency_gap.rename('heuristics_v2')
    
    return heuristics_matrix
