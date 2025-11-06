import pandas as pd

def heuristics_v2(df):
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    momentum_5 = close.pct_change(5)
    momentum_10 = close.pct_change(10)
    momentum_20 = close.pct_change(20)
    
    acceleration = momentum_5 - momentum_10
    volatility = (high - low).rolling(window=20).std()
    vol_adjusted_momentum = momentum_20 / (volatility + 1e-8)
    
    volume_trend = volume.rolling(window=10).mean() / volume.rolling(window=30).mean()
    
    heuristics_matrix = acceleration * vol_adjusted_momentum * volume_trend
    return heuristics_matrix
