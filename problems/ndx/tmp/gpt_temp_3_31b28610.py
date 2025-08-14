def heuristics_v2(df):
    sma_50 = df['close'].rolling(window=50).mean()
    sma_200 = df['close'].rolling(window=200).mean()
    ma_diff = sma_50 - sma_200
    volume_log_change = (df['volume'] / df['volume'].shift(10)).apply(np.log)
    heuristic_values = ma_diff * volume_log_change
    heuristics_matrix = heuristic_values.ewm(span=20, adjust=False).mean().dropna()
    return heuristics_matrix
