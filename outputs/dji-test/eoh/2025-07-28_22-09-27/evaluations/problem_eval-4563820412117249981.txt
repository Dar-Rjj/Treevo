import pandas as pd

def heuristics_v2(df):
    log_diff = (df['close'].apply(np.log) - df['open'].apply(np.log))
    vol_std = df['volume'].rolling(window=10).std()
    mom_high = (df['high'] - df['high'].shift(5)) / df['high'].shift(5)
    heuristics_matrix = (log_diff / vol_std) * mom_high
    return heuristics_matrix
