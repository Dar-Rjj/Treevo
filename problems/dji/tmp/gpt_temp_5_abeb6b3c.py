import pandas as pd

def heuristics_v2(df):
    ema_close = df['close'].ewm(span=10, adjust=False).mean()
    ema_volume = df['volume'].ewm(span=10, adjust=False).mean()
    mad_close = df['close'].rolling(window=10).apply(lambda x: (x - x.median()).abs().median(), raw=True)
    heuristics_matrix = (ema_close - ema_volume) / mad_close
    return heuristics_matrix
