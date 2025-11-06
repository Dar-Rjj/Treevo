import pandas as pd

def heuristics_v2(df):
    short_trend = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    medium_trend = (df['close'] - df['close'].shift(8)) / df['close'].shift(8)
    
    trend_convergence = short_trend * medium_trend
    
    volume_persistence = df['volume'].rolling(window=10).apply(lambda x: (x > x.shift(1)).sum() / len(x), raw=False)
    
    volatility_regime = (df['high'] - df['low']).rolling(window=5).std() / (df['high'] - df['low']).rolling(window=15).std()
    
    heuristics_matrix = trend_convergence * volume_persistence * volatility_regime
    
    return heuristics_matrix
