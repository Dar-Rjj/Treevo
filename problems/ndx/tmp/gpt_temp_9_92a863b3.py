def heuristics_v2(df):
    log_return = df['close'].pct_change().apply(lambda x: np.log(1 + x))
    high_low_ratio = (df['high'] / df['low']).rolling(window=5).mean()
    heuristic_values = log_return * high_low_ratio
    heuristics_matrix = heuristic_values.ewm(span=10, adjust=False).mean().dropna()
    return heuristics_matrix
