def heuristics_v2(df):
    daily_returns = df['close'].pct_change()
    cum_return_20 = (1 + daily_returns).rolling(window=20).apply(np.prod, raw=True) - 1
    std_dev_20 = daily_returns.rolling(window=20).std()
    heuristics_matrix = cum_return_20 / std_dev_20
    return heuristics_matrix
