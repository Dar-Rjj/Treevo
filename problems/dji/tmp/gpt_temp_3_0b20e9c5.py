def heuristics_v2(df):
    log_returns = np.log(df['close'] / df['close'].shift(1))
    step1 = log_returns.rolling(window=5).mean()
    step2 = df['close'].rolling(window=20).std()
    step3 = df['volume'] / df['volume'].shift(10)
    heuristics_matrix = (step1 + step2) * step3
    return heuristics_matrix
