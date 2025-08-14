def heuristics_v2(df):
    def ema_diff(price, fast=15, slow=45):
        ema_fast = price.ewm(span=fast, adjust=False).mean()
        ema_slow = price.ewm(span=slow, adjust=False).mean()
        return ema_fast - ema_slow

    def roc(series, periods=30):
        return (series - series.shift(periods)) / series.shift(periods)

    ema_signal = ema_diff(df['close'])
    roc_volume = roc(df['volume'])
    combined_factor = (ema_signal + roc_volume).rename('combined_factor')
    weights = np.arange(1, 31)
    heuristics_matrix = combined_factor.rolling(window=30).apply(lambda x: np.average(x, weights=weights), raw=True).rename('heuristic_factor')

    return heuristics_matrix
