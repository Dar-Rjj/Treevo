import pandas as pd

def heuristics_v2(df):
    price_momentum = (df['close'] - df['close'].shift(8)) / df['close'].shift(8)
    volume_trend = df['volume'].rolling(window=12).apply(lambda x: pd.Series(x).pct_change().mean(), raw=False)
    volatility_regime = (df['high'] - df['low']).rolling(window=5).std() / (df['high'] - df['low']).rolling(window=15).std()
    
    heuristics_matrix = price_momentum * volume_trend * volatility_regime
    heuristics_matrix = heuristics_matrix.rolling(window=6).mean()
    
    return heuristics_matrix
