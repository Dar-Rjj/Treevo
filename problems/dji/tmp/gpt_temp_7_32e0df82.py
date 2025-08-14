def heuristics_v2(df):
    log_returns = (df['close'] / df['close'].shift(1)).apply(np.log)
    step1 = log_returns.rolling(window=20).mean()
    step2 = df['volume'].rolling(window=20).std()
    heuristics_matrix = step1 * step2
    return heuristics_matrix
