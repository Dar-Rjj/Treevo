def heuristics_v2(df):
    log_returns = df['close'].pct_change().apply(lambda x: np.log(1 + x))
    daily_range_ratio = (df['high'] - df['low']) / df['close']
    heuristics_matrix = (log_returns * daily_range_ratio).rolling(window=20).mean()
    return heuristics_matrix
