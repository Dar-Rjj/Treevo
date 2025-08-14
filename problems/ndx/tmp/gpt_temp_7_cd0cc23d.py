def heuristics_v2(df):
    log_returns = np.log(df['close'] / df['close'].shift(1))
    ma_5 = log_returns.rolling(window=5).mean()
    ma_20 = log_returns.rolling(window=20).mean()
    heuristic_values = ma_20 - ma_5
    heuristics_matrix = heuristic_values.ewm(span=7, adjust=False).mean().dropna()
    return heuristics_matrix
