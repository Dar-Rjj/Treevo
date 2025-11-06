import pandas as pd
import numpy as np

def heuristics_v2(df):
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    
    # Price oscillation extremes relative to recent range
    mid_price = (high + low) / 2
    price_oscillation = (close - mid_price) / (high - low + 1e-8)
    oscillation_extreme = price_oscillation.rolling(6).apply(lambda x: (x[-1] - x.mean()) / (x.std() + 1e-8))
    
    # Volume divergence from price movement
    price_movement = close.pct_change(3)
    volume_trend = volume.rolling(8).apply(lambda x: (x[-1] - x.mean()) / (x.std() + 1e-8))
    volume_divergence = np.sign(price_movement) * volume_trend
    
    # Volatility regime detection for mean-reversion strength
    volatility_regime = (high - low).rolling(10).std() / close
    regime_threshold = volatility_regime.rolling(20).quantile(0.6)
    regime_filter = np.where(volatility_regime > regime_threshold, 1.5, 0.8)
    
    # Composite mean-reversion factor
    heuristics_matrix = oscillation_extreme * volume_divergence * regime_filter
    
    return heuristics_matrix
