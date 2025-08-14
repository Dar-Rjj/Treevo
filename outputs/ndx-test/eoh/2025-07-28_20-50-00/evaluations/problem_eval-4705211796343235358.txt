def heuristics_v2(df):
    ewma_50 = df['close'].ewm(span=50).mean()
    sma_20 = df['close'].rolling(window=20).mean()
    log_returns = np.log(df['close'] / df['close'].shift(1))
    std_30 = log_returns.rolling(window=30).std()
    heuristics_matrix = (ewma_50 - sma_20) / std_30
    return heuristics_matrix
