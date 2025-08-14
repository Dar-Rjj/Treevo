def heuristics_v4(df):
    ma_close = df['close'].rolling(window=5).mean()
    ma_volume = df['volume'].rolling(window=10).mean()
    heuristics_matrix = ma_close * ma_volume
    return heuristics_matrix
