def heuristics_v2(df):
    def ema_diff(price, fast=12, slow=60):
        ema_fast = price.ewm(span=fast, adjust=False).mean()
        ema_slow = price.ewm(span=slow, adjust=False).mean()
        return ema_fast - ema_slow

    def roc(series, periods=25):
        return series.pct_change(periods=periods)

    def wma(series, window=35):
        weights = np.arange(1, window + 1)
        return series.rolling(window=window).apply(lambda x: (x * weights).sum() / weights.sum(), raw=True)

    ema_signal = ema_diff(df['close'])
    roc_high = roc(df['high'])
    combined_factor = (ema_signal + roc_high).rename('combined_factor')
    heuristics_matrix = wma(combined_factor).rename('heuristic_factor')

    return heuristics_matrix
