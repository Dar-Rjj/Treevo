import pandas as pd

def heuristics_v2(df):
    ema_ratio = df['high'] / df['low']
    ema = ema_ratio.ewm(span=10, adjust=False).mean()
    heuristics_matrix = (ema * np.log(df['volume'])).dropna()
    return heuristics_matrix
