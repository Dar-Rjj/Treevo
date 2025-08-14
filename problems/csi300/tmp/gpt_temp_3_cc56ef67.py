import pandas as pd

def heuristics_v2(df):
    close_return = df['close'].pct_change(10)
    volume_roc = df['volume'].pct_change(5)
    rsi = lambda x: 100 - (100 / (1 + (x.diff().clip(lower=0).ewm(alpha=1/14).mean() / -x.diff().clip(upper=0).ewm(alpha=1/14).mean())))
    close_rsi = df['close'].rolling(window=14).apply(rsi, raw=True)
    heuristics_matrix = (close_return + volume_roc) * (1 + close_rsi / 100)
    return heuristics_matrix
