def heuristics_v2(df):
    sma_5 = df['close'].rolling(window=5).mean()
    sma_20 = df['close'].rolling(window=20).mean()
    price_diff = sma_5 - sma_20
    volume_log_return = np.log(df['volume'] / df['volume'].shift(1))
    heuristic_values = price_diff * volume_log_return
    heuristics_matrix = heuristic_values.ewm(span=14, adjust=False).mean().dropna()
    return heuristics_matrix
