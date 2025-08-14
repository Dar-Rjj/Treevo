def heuristics_v2(df):
    short_wma = df['close'].rolling(window=15).apply(lambda x: (x * pd.Series(range(1, len(x) + 1))).sum() / pd.Series(range(1, len(x) + 1)).sum(), raw=True)
    long_wma = df['close'].rolling(window=30).apply(lambda x: (x * pd.Series(range(1, len(x) + 1))).sum() / pd.Series(range(1, len(x) + 1)).sum(), raw=True)
    pv_momentum_smoothed = (df['close'] * df['volume']).rolling(window=10).sum() / df['volume'].rolling(window=10).sum()
    wma_diff = short_wma - long_wma
    heuristics_matrix = (wma_diff + pv_momentum_smoothed).rank(pct=True)
    return heuristics_matrix
