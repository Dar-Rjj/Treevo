def heuristics_v2(df):
    window_size = 20
    df['ema_close'] = df['close'].ewm(span=window_size, adjust=False).mean()
    df['ema_volume'] = df['volume'].ewm(span=window_size, adjust=False).mean()
    std_close = df['close'].rolling(window=window_size).std()
    std_volume = df['volume'].rolling(window=window_size).std()
    weight_close_ratio = std_close / (std_close + std_volume)
    heuristics_matrix = (df['close'] / df[['high']].rolling(window=window_size).max()) * weight_close_ratio + df['ema_volume'] * (1 - weight_close_ratio)
    return heuristics_matrix
