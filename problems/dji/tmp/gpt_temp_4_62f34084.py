def heuristics_v4(df):
    ema_close = df['close'].ewm(span=10, adjust=False).mean()
    sum_volume = df['volume'].rolling(window=5).sum()
    heuristics_matrix = ema_close / sum_volume
    return heuristics_matrix
