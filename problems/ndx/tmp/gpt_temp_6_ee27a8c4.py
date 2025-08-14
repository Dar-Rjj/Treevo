def heuristics_v2(df):
    df['daily_return'] = df['close'].pct_change()
    ewma_volatility = df['daily_return'].ewm(span=10, min_periods=20).std()
    return heuristics_matrix
