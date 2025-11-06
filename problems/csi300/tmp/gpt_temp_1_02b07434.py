import pandas as pd

def heuristics_v2(df):
    price_momentum_acceleration = (df['close'] / df['close'].shift(5) - 1) - (df['close'].shift(5) / df['close'].shift(10) - 1)
    volume_trend_persistence = df['volume'].rolling(window=10).apply(lambda x: pd.Series(x).autocorr(lag=1))
    volatility_breakout = (df['high'] - df['low']) / (df['high'].rolling(window=10).max() - df['low'].rolling(window=10).min())
    
    heuristics_matrix = price_momentum_acceleration * volume_trend_persistence * volatility_breakout
    heuristics_matrix = heuristics_matrix.rolling(window=8).mean()
    
    return heuristics_matrix
