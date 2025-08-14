def heuristics_v2(df):
    def compute_momentum(df, window=20):
        return df['close'].pct_change(window)

    def compute_rsi(df, window=14):
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def compute_atr(df, window=14):
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(window).mean()

    def compute_trading_volume_indicator(df, window=20):
        return df['volume'].pct_change(window)

    momentum = compute_momentum(df)
    rsi = compute_rsi(df)
    atr = compute_atr(df)
    tvi = compute_trading_volume_indicator(df)

    heuristics_matrix = (momentum + rsi - atr) * tvi
    return heuristics_matrix
