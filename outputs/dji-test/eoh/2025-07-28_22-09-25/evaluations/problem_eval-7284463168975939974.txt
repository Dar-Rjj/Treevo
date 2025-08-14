def heuristics_v2(df):
    def rsi(series, periods=14):
        delta = series.diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    rsi_15 = rsi(df['close'], periods=15)
    high_30 = df['high'].rolling(window=30).max()
    low_30 = df['low'].rolling(window=30).min()
    price_range = high_30 - low_30
    volume_ema_90 = df['volume'].ewm(span=90, adjust=False).mean()
    volume_ratio_ln = np.log(df['volume'] / volume_ema_90)
    
    heuristics_matrix = rsi_15 * price_range * volume_ratio_ln
    return heuristics_matrix
