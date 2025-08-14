import pandas as pd

def heuristics_v2(df):
    # Dynamic window size for WMA based on volume
    window_size = (df['volume'].rank(pct=True) * 10).astype(int)
    wma_close = df['close'].rolling(window=window_size, min_periods=1).apply(lambda x: (x * range(1, len(x)+1)).sum() / sum(range(1, len(x)+1)), raw=False)
    sma_low = df['low'].rolling(window=5, min_periods=1).mean()
    heuristics_matrix = wma_close - sma_low
    return heuristics_matrix
