def heuristics_v4(df):
    ema_close_10d = df['close'].ewm(span=10, adjust=False).mean()
    std_volume_10d = df['volume'].rolling(window=10).std()
    mean_volume_10d = df['volume'].rolling(window=10).mean()
    coef_variation_volume = std_volume_10d / mean_volume_10d
    heuristics_matrix = ema_close_10d / coef_variation_volume
    return heuristics_matrix
