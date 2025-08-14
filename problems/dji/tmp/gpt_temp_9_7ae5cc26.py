import pandas as pd

def heuristics_v2(df):
    ema_close_5 = df['close'].ewm(span=5, adjust=False).mean()
    ema_open_5 = df['open'].ewm(span=5, adjust=False).mean()
    ema_close_15 = df['close'].ewm(span=15, adjust=False).mean()
    ema_open_15 = df['open'].ewm(span=15, adjust=False).mean()
    diff_5 = ema_close_5 - ema_open_5
    diff_15 = ema_close_15 - ema_open_15
    vol_log = df['volume'].apply(lambda x: 0 if x <= 0 else math.log(x))
    heuristics_matrix = (diff_5 * diff_15) * vol_log
    return heuristics_matrix
