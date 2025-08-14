def heuristics_v2(df):
    sma_volume = df['volume'].rolling(window=5).mean()
    log_diff_open_close = (df['open'] - df['close']).apply(lambda x: abs(x)).apply(np.log)
    heuristics_matrix = log_diff_open_close / sma_volume
    return heuristics_matrix
