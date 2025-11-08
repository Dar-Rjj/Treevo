import pandas as pd
import numpy as np

def heuristics_v2(df):
    close = df['close']
    volume = df['volume']
    high = df['high']
    low = df['low']
    
    volume_trend = volume.rolling(window=15).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    price_residual = close - (close.rolling(window=10).mean() + volume_trend * 0.1)
    residual_momentum = price_residual.rolling(window=5).mean()
    
    volatility_regime = (high - low).rolling(window=20).std() / (high - low).rolling(window=60).std()
    price_position = (close - low.rolling(window=10).min()) / (high.rolling(window=10).max() - low.rolling(window=10).min())
    regime_adjusted_reversion = (0.5 - price_position) * volatility_regime
    
    volume_accel = volume.rolling(window=8).apply(lambda x: np.polyfit(range(len(x)), x, 2)[0])
    volume_confirmation = np.sign(volume_accel) * volume_accel.rolling(window=5).std()
    
    heuristics_matrix = residual_momentum * regime_adjusted_reversion * volume_confirmation
    return heuristics_matrix
