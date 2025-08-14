def heuristics_v2(df):
    log_returns = np.log(df['close']).diff().rolling(window=10).sum()
    smoothed_matrix = log_returns.rolling(window=20).median()
    heuristics_matrix = smoothed_matrix.rolling(window=30).skew()
    return heuristics_matrix
