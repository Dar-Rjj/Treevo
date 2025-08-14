def heuristics_v2(df):
    def ema_diff(price, fast=10, slow=30):
        ema_fast = price.ewm(span=fast, adjust=False).mean()
        ema_slow = price.ewm(span=slow, adjust=False).mean()
        return ema_fast - ema_slow

    def pct_change(series, periods=1):
        return series.pct_change(periods)

    ema_signal = ema_diff(df['high'])
    pct_change_volume = pct_change(df['volume'])
    combined_factor = (ema_signal + pct_change_volume).rename('combined_factor')
    weights = np.arange(1, 31)  # Weights for the WMA
    heuristics_matrix = combined_factor.rolling(window=30).apply(lambda x: np.dot(x, weights)/weights.sum(), raw=True).rename('heuristic_factor')

    return heuristics_matrix
