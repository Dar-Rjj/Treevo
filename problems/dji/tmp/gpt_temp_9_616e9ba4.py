def heuristics_v2(df):
    log_returns = np.log(df['close'] / df['close'].shift(1))
    log_returns_5d_sum = log_returns.rolling(window=5).sum()
    heuristics_matrix = (log_returns_5d_sum - log_returns_5d_sum.rolling(window=20).mean()) / log_returns_5d_sum.rolling(window=20).std()
    return heuristics_matrix
