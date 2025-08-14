import pandas as pd

def heuristics_v2(df):
    ma_close_10 = df['close'].rolling(window=10).mean()
    cum_volume_60 = df['volume'].rolling(window=60).sum()
    log_ma_close_10 = ma_close_10.apply(lambda x: math.log(x) if x > 0 else 0)
    log_cum_volume_60 = cum_volume_60.apply(lambda x: math.log(x) if x > 0 else 0)
    roc_high_5 = df['high'].pct_change(periods=5)
    heuristics_matrix = log_ma_close_10 - log_cum_volume_60 + roc_high_5
    return heuristics_matrix
