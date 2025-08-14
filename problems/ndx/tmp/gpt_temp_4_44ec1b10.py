def heuristics_v2(df):
    def calculate_roc(column, n):
        return (column - column.shift(n)) / column.shift(n)

    def calculate_rsi(close, period=14):
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_atr(high, low, close, period=14):
        high_low = high - low
        high_close = (high - close.shift()).abs()
        low_close = (low - close.shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = ranges.rolling(window=period).mean()
        return atr

    roc_short = calculate_roc(df['close'], 5)
    roc_medium = calculate_roc(df['close'], 21)
    roc_long = calculate_roc(df['close'], 63)
    rsi = calculate_rsi(df['close'])
    atr = calculate_atr(df['high'], df['low'], df['close'])

    heuristics_matrix = 0.2 * roc_short + 0.2 * roc_medium + 0.2 * roc_long + 0.2 * rsi + 0.2 * atr
    return heuristics_matrix
