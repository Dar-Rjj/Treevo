def heuristics_v4(df):
    ema_high_30d = df['high'].ewm(span=30, adjust=False).mean()
    cum_volume_120d = df['volume'].rolling(window=120).sum()
    heuristics_matrix = ema_high_30d / cum_volume_120d
    return heuristics_matrix
