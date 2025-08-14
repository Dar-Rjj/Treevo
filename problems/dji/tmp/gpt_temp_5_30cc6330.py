def heuristics_v2(df):
    pct_change_close = df['close'].pct_change()
    log_change_volume = df['volume'].apply(np.log).diff()
    intermediate_matrix = (pct_change_close * log_change_volume).rolling(window=10).sum()
    heuristics_matrix = intermediate_matrix.ewm(span=5).mean()
    return heuristics_matrix
