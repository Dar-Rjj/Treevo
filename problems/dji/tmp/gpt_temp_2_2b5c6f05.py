def heuristics_v2(df):
    def ema_diff(price, fast=12, slow=26):
        ema_fast = price.ewm(span=fast, adjust=False).mean()
        ema_slow = price.ewm(span=slow, adjust=False).mean()
        return ema_fast - ema_slow

    def roc(series, periods=10):
        return series.pct_change(periods)

    ema_signal = ema_diff(df['close'])
    roc_adj_close = roc(df['adj_close'])
    combined_factor = (ema_signal + roc_adj_close).rename('combined_factor')
    heuristics_matrix = combined_factor.rolling(window=30).mean().rename('heuristic_factor')

    return heuristics_matrix
