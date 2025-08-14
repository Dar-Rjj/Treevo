def heuristics_v2(df):
    log_diff = (df['high'] - df['low']).apply(np.log)
    log_diff_sum = log_diff.rolling(window=10).sum()
    avg_volume = df['volume'].rolling(window=10).mean()
    heuristics_matrix = log_diff_sum / avg_volume
    return heuristics_matrix
