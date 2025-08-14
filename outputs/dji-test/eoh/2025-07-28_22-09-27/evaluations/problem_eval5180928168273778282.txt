def heuristics_v2(df):
    adj_close_avg_high_low_ratio = df['adj_close'] / ((df['high'] + df['low']) / 2)
    heuristics_matrix = adj_close_avg_high_low_ratio.ewm(span=20, adjust=False).mean()
    return heuristics_matrix
