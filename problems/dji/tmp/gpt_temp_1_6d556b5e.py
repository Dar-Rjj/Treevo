import pandas as pd

def heuristics_v2(df):
    df['SMA10'] = df['close'].rolling(window=10).mean()
    df['EMA50_vol'] = df['volume'].ewm(span=50, adjust=False).mean()
    df['ATR14'] = (df['high'].rolling(window=14).max() - df['low'].rolling(window=14).min()).apply(lambda x: max(x, 1))
    heuristics_matrix = (df['SMA10'] / df['EMA50_vol']) * df['ATR14'].apply(np.log)
    return heuristics_matrix
