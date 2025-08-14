import pandas as pd

def heuristics_v2(df):
    log_diff = df['high'].apply(lambda x: np.log(x)) - df['low'].apply(lambda x: np.log(x))
    sma_20_close = df['close'].rolling(window=20).mean()
    roc_volume = df['volume'] / df['volume'].shift(5) - 1
    heuristics_matrix = (log_diff * sma_20_close) * roc_volume
    return heuristics_matrix
