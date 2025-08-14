import pandas as pd

def heuristics_v2(df):
    def ema_line(price, fast=15, slow=40):
        ema_fast = price.ewm(span=fast, adjust=False).mean()
        ema_slow = price.ewm(span=slow, adjust=False).mean()
        return ema_fast - ema_slow

    def aroc(series, periods=25):
        return series.diff(periods=periods).abs()

    def wma_line(series, window=18):
        weights = pd.Series(range(1,window+1))
        return series.rolling(window=window).apply(lambda x: (x*weights).sum() / weights.sum(), raw=True)

    ema_signal = ema_line(df['adj_close'])
    aroc_volume = aroc(df['volume'])
    combined_factor = (ema_signal + aroc_volume).rename('combined_factor')
    heuristics_matrix = wma_line(combined_factor).rename('heuristic_factor')

    return heuristics_matrix
