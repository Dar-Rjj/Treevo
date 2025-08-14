def heuristics_v2(df):
    def ema_diff(price, fast=10, slow=50):
        ema_fast = price.ewm(span=fast, adjust=False).mean()
        ema_slow = price.ewm(span=slow, adjust=False).mean()
        return ema_fast - ema_slow

    def roc(series, periods=20):
        return (series - series.shift(periods)) / series.shift(periods)

    ema_signal = ema_diff(df['close'])
    roc_high = roc(df['high'])
    combined_factor = (ema_signal + roc_high).rename('combined_factor')
    weights = np.arange(1, 31)  # Weights for the WMA
    heuristics_matrix = (combined_factor * weights).rolling(window=30).sum() / weights.sum()

    return heuristics_matrix
