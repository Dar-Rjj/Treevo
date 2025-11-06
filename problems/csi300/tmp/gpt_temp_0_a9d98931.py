import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    amount = df['amount']
    
    # Volatility regime detection using rolling percentiles
    vol_30 = close.pct_change().rolling(30).std()
    vol_regime = vol_30.rolling(90).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    # Momentum breakout component (high volatility regime)
    breakout_signal = (close - close.rolling(20).mean()) / close.rolling(20).std()
    
    # Volume-scaled mean reversion (low volatility regime)
    vwap = amount / (volume + 1e-8)
    price_deviation = (close - vwap) / close.rolling(10).std()
    volume_scale = volume.rolling(5).mean() / volume.rolling(20).mean()
    mean_reversion_signal = -price_deviation * volume_scale
    
    # Regime-dependent adaptive weighting
    regime_weight = np.tanh(vol_regime * 3)
    adaptive_factor = breakout_signal * regime_weight + mean_reversion_signal * (1 - regime_weight)
    
    # Volume confirmation filter
    volume_accel = volume.rolling(5).mean() / volume.rolling(20).mean() - 1
    heuristics_matrix = adaptive_factor * np.sign(volume_accel)
    
    return heuristics_matrix
