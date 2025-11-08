import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    trend_persistence = close.rolling(window=15).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    liquidity_volatility = volume.rolling(window=20).std() / volume.rolling(window=20).mean()
    trend_signal = trend_persistence / liquidity_volatility
    
    market_regime = close.rolling(window=30).std() / close.rolling(window=30).mean()
    price_residual = close - close.rolling(window=10).mean()
    regime_adjusted_residual = price_residual / market_regime
    
    volume_acceleration = volume.pct_change(5)
    reversal_signal = -regime_adjusted_residual * volume_acceleration
    
    heuristics_matrix = trend_signal * 0.7 + reversal_signal * 0.3
    
    return heuristics_matrix
