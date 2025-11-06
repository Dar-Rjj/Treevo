import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    returns = close.pct_change()
    momentum_persistence = returns.rolling(10).apply(lambda x: x.autocorr(), raw=False)
    
    volume_acceleration = volume.rolling(5).mean() / volume.rolling(20).mean() - 1
    volatility_measure = (high - low).rolling(10).std() / close.rolling(10).mean()
    volume_factor = volume_acceleration / (volatility_measure + 1e-8)
    
    trend_regime = (close.rolling(5).mean() - close.rolling(20).mean()) / close.rolling(20).std()
    regime_weight = np.where(trend_regime > 0, 1 + np.tanh(trend_regime), 1 - np.tanh(trend_regime))
    
    heuristics_matrix = momentum_persistence * volume_factor * regime_weight
    
    return heuristics_matrix
