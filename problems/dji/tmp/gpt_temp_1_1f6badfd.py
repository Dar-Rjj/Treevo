def heuristics_v2(df):
    df['price_ema'] = df['close'].ewm(span=20, adjust=False).mean()
    df['cum_volume'] = df['volume'].rolling(window=60).sum()
    df['atr'] = df[['high', 'low', 'close']].rolling(window=30).apply(
        lambda x: max(x[0] - x[1], x[1] - x[2], x[0] - x[2]), raw=True)
    heuristics_matrix = (df['price_ema'].pct_change().shift(-1) * df['cum_volume']) / df['atr']
    return heuristics_matrix
