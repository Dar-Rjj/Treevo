def heuristics_v2(df):
    volume_price_ratio = df['volume'] / df['adj_close']
    heuristics_matrix = volume_price_ratio.ewm(span=20, adjust=False).mean()
    return heuristics_matrix
