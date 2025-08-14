def heuristics_v2(df):
    def ema_diff(price, fast=5, slow=20):
        ema_fast = price.ewm(span=fast, adjust=False).mean()
        ema_slow = price.ewm(span=slow, adjust=False).mean()
        return ema_fast - ema_slow

    def close_mad(price, window=50):
        rolling_mad = price.rolling(window=window).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        return rolling_mad / price

    ema_signal = ema_diff(df['close'])
    mad_signal = close_mad(df['close'])
    combined_factor = (ema_signal + mad_signal).rename('combined_factor')
    heuristics_matrix = combined_factor.rolling(window=15).mean().rename('heuristic_factor')

    return heuristics_matrix
