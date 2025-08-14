def heuristics_v2(df):
    roc = df['close'].pct_change(periods=14).fillna(0)
    sma_50 = df['close'].rolling(window=50).mean().fillna(0)
    distance_to_sma = df['close'] - sma_50
    heuristics_matrix = roc * (np.log(1 + abs(distance_to_sma)) + 1)
    return heuristics_matrix
