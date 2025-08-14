def heuristics_v2(df):
    pct_change_close = df['close'].pct_change()
    pct_change_volume = df['volume'].pct_change().apply(lambda x: np.log(1 + x))
    intermediate_matrix = (pct_change_close + pct_change_volume).rolling(window=10).sum()
    heuristics_matrix = intermediate_matrix.ewm(span=5).mean()
    return heuristics_matrix
