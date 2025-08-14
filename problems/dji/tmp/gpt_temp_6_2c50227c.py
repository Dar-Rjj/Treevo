def heuristics_v2(df):
    daily_return = df['close'].pct_change()
    sma_5_days = daily_return.rolling(window=5).mean()
    log_returns = np.log(df['close'] / df['close'].shift(1))
    std_10_days = log_returns.rolling(window=10).std()
    heuristics_matrix = 0.7 * sma_5_days + 0.3 * std_10_days
    return heuristics_matrix
