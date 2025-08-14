def heuristics_v2(df):
    ma_20_volume = df['volume'].rolling(window=20).mean()
    max_30_adj_close = df['adj_close'].rolling(window=30).max()
    heuristics_matrix = ma_20_volume / max_30_adj_close
    return heuristics_matrix
