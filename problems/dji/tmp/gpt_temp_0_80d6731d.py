def heuristics_v2(df):
    def atr(df, n=14):
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(window=n).mean()

    def macd(df, fast=12, slow=26, signal=9):
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line - signal_line

    atr_value = atr(df)
    macd_value = macd(df)
    heuristics_matrix = (atr_value + macd_value) / 2
    return heuristics_matrix
