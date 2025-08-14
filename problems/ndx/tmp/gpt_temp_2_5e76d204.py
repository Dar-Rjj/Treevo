def heuristics_v4(df):
    avg_price = (df['high'] + df['low']) / 2
    adj_close_vs_avg = df['close'] - avg_price
    vol_std = df['volume'].rolling(window=20).std()
    heuristic_values = adj_close_vs_avg / vol_std
    heuristics_matrix = heuristic_values.ewm(span=7, adjust=False).mean().dropna()
    return heuristics_matrix
