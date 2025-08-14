def heuristics_v4(df):
    log_returns = np.log(df['close'] / df['close'].shift(1))
    heuristics_matrix = log_returns.rolling(window=30).sum().dropna()
    return heuristics_matrix
