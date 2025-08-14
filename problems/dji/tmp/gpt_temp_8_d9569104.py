import pandas as pd

def heuristics_v2(df):
    df['Close_MA10'] = df['close'].rolling(window=10).mean()
    df['Volume_MA5'] = df['volume'].rolling(window=5).mean()
    df['High_MA3'] = df['high'].rolling(window=3).mean()
    heuristics_matrix = (df['close'] / df['Close_MA10']) * (df['volume'] / df['Volume_MA5']).apply(lambda x: x if x > 0 else 1).apply(np.log) - df['High_MA3']
    return heuristics_matrix
