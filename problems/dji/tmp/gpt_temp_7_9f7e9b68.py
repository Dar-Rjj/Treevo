import pandas as pd

def heuristics_v2(df):
    nvi = ((df['close'].diff() / df['close'].shift()) * (df['volume'] < df['volume'].shift()).astype(int)).cumsum()
    wma_nvi = (nvi * list(range(1, 21))).rolling(window=20).sum() / 210
    sd_close = df['close'].rolling(window=7).std()
    heuristics_matrix = (wma_nvi * sd_close).fillna(0)
    return heuristics_matrix
