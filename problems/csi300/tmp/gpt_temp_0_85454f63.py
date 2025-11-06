import pandas as pd

def heuristics_v2(df):
    intraday_return = (df['close'] - df['open']) / df['open']
    price_volatility = (df['high'] - df['low']).rolling(window=10).std()
    volume_persistence = df['volume'].rolling(window=10).apply(lambda x: (x > x.mean()).sum())
    
    raw_factor = -intraday_return * volume_persistence / (price_volatility + 1e-8)
    heuristics_matrix = raw_factor.rename('heuristics_v2')
    
    return heuristics_matrix
