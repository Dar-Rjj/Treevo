def heuristics_v2(df):
    def compute_heuristic(row):
        ma_short = row['close'].rolling(window=5).mean()
        ma_long = row['close'].rolling(window=20).mean()
        vol_change = (row['volume'] - row['volume'].shift(1)) / row['volume'].shift(1)
        return (ma_short - ma_long) * vol_change

    heuristics_matrix = df.apply(compute_heuristic, axis=1)
    return heuristics_matrix
