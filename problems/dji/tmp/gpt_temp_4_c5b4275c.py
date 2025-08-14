def heuristics_v2(df):
    def rsi(series, periods=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def atr(df, periods=14):
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(window=periods).mean()

    rsi_close = rsi(df['close'])
    atr_val = atr(df)
    ema_rsi = rsi_close.ewm(span=10, adjust=False).mean()
    ema_atr = atr_val.ewm(span=10, adjust=False).mean()
    combined_factor = (ema_rsi + ema_atr).rename('combined_factor')
    heuristics_matrix = combined_factor.ewm(span=18, adjust=False).mean().rename('heuristic_factor')

    return heuristics_matrix
