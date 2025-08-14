def heuristics_v2(df):
    window_size = 20
    df['ema_close'] = df['close'].ewm(span=window_size, adjust=False).mean()
    positive_changes = df['close'].pct_change().apply(lambda x: x if x > 0 else 0).rolling(window=window_size).mean()
    negative_changes = df['close'].pct_change().apply(lambda x: x if x < 0 else 0).rolling(window=window_size).mean()
    log_diff = (df['close'] - df['ema_close']).apply(np.log)
    heuristics_matrix = (positive_changes + negative_changes) * log_diff
    return heuristics_matrix
