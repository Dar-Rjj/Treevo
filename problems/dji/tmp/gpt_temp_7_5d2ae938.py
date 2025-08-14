import pandas as pd

def heuristics_v2(df):
    df['Close_MA5'] = df['close'].rolling(window=5).mean()
    df['Volume_MA20'] = df['volume'].rolling(window=20).mean()
    heuristics_matrix = df['Close_MA5'] + pd.Series(np.log(df['volume'] / df['Volume_MA20']))
    return heuristics_matrix
