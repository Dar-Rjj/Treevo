def heuristics_v2(df):
    pct_change_close = df['close'].pct_change()
    pct_change_volume = df['volume'].pct_change()
    intermediate_matrix = (pct_change_close * pct_change_volume).rolling(window=10).sum()
    weights = np.arange(1, 6)
    heuristics_matrix = intermediate_matrix.rolling(window=5).apply(lambda x: (x * weights).sum() / weights.sum(), raw=True)
    return heuristics_matrix
