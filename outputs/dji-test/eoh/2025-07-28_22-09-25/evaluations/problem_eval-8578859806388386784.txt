def heuristics_v2(df):
    sum_30_volume = df['volume'].rolling(window=30).sum()
    sum_90_high_adj_close = (df['high'] + df['adj_close']).rolling(window=90).sum()
    heuristics_matrix = sum_30_volume / sum_90_high_adj_close
    return heuristics_matrix
