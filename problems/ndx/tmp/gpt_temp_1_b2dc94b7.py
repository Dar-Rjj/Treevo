def heuristics_v2(df):
    median_price = (df['high'] + df['low']) / 2
    ema_close = df['close'].ewm(span=14, adjust=False).mean()
    heuristic_values = median_price - ema_close
    weights = np.arange(1, 21)
    heuristics_matrix = heuristic_values.rolling(window=20).apply(lambda x: np.average(x, weights=weights), raw=True).dropna()
    return heuristics_matrix
