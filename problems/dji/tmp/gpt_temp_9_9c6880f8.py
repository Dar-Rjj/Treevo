import pandas as pd

def heuristics_v2(df):
    def sma(price, window=20):
        return price.rolling(window=window).mean()

    def log_change(series):
        return series.apply(lambda x: 0 if x == 0 else (x - series.shift(1)) / series.shift(1)).fillna(0)

    def wma(series, window=10):
        weights = pd.Series(range(1, window + 1))
        return series.rolling(window=window).apply(lambda prices: (prices * weights).sum() / weights.sum(), raw=True)

    price_to_sma_ratio = df['close'] / sma(df['close'])
    volume_log_change = log_change(df['volume'])
    combined_factor = (price_to_sma_ratio * volume_log_change).rename('combined_factor')
    heuristics_matrix = wma(combined_factor).rename('heuristic_factor')

    return heuristics_matrix
