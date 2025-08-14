def heuristics_v2(df):
    log_return = np.log(df['close']).diff()
    sma_volume_5 = df['volume'].rolling(window=5).mean()
    heuristics_matrix = log_return * sma_volume_5
    return heuristics_matrix
