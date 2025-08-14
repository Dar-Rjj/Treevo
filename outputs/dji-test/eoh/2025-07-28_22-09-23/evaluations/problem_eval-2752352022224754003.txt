def heuristics_v2(df):
    def calculate_heuristic(row, avg_close):
        return abs(row['close'] - avg_close)
    
    avg_close = df['close'].rolling(window=10).mean()
    df['heuristic'] = df.apply(calculate_heuristic, args=(avg_close,), axis=1)
    heuristics_matrix = df['heuristic'].rolling(window=10).sum() / df['close'].rolling(window=10).std()
    return heuristics_matrix
