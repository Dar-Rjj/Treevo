import pandas as pd

def heuristics_v2(df):
    price_diff = df['high'] - df['low']
    log_ret = df['close'].pct_change().apply(lambda x: x if x == 0 else math.log(1 + x))
    volume_wma = df['volume'].rolling(window=5).apply(lambda x: (x * range(1, len(x) + 1)).sum() / x.sum(), raw=True)
    heuristics_matrix = price_diff + (volume_wma * log_ret)
    return heuristics_matrix
