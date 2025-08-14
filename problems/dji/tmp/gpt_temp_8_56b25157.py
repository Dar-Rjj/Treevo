def heuristics_v2(df):
    log_change_close = df['close'].apply(np.log).diff()
    log_change_volume = df['volume'].apply(np.log).diff()
    intermediate_matrix = (log_change_close + log_change_volume).rolling(window=15).sum()
    heuristics_matrix = intermediate_matrix.rolling(window=10).std()
    return heuristics_matrix
