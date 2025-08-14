import pandas as pd

def heuristics_v2(df):
    ema_30_high = df['high'].ewm(span=30, adjust=False).mean()
    wma_60_volume = weighted_moving_average(df['volume'], window=60)
    heuristics_matrix = ema_30_high - wma_60_volume
    return heuristics_matrix

def weighted_moving_average(series, window=30):
    weights = pd.Series(range(1, window+1))
    wma = series.rolling(window=window).apply(lambda x: (x * weights).sum() / weights.sum(), raw=True)
    return heuristics_matrix
