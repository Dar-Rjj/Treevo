def heuristics_v2(df):
    def ema_diff(price, fast=12, slow=26):
        ema_fast = price.ewm(span=fast, adjust=False).mean()
        ema_slow = price.ewm(span=slow, adjust=False).mean()
        return ema_fast - ema_slow

    def rsi(series, periods=14):
        delta = series.diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    ema_signal = ema_diff(df['close'])
    rsi_close = rsi(df['adj_close'], periods=14)
    combined_factor = (ema_signal + rsi_close).rename('combined_factor')
    heuristics_matrix = combined_factor.rolling(window=20).std().rename('heuristic_factor')

    return heuristics_matrix
