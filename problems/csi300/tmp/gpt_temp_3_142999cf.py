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
    
    reversal_signal = acceleration.rolling(window=5).mean() - vol_adjusted_momentum
    
    volume_trend = volume.rolling(window=10).mean() / volume.rolling(window=30).mean()
    volume_filter = (volume_trend > 1).astype(int)
    
    heuristics_matrix = reversal_signal * volume_filter
    heuristics_matrix.name = 'heuristics_v2'
    
    return heuristics_matrix
