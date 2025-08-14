def heuristics_v2(df):
    def calc_heuristic(row):
        return (row['high'] - row['low']) / row['volume']

    df['heuristics'] = df.rolling(window=5).apply(calc_heuristic, raw=False)
    heuristics_matrix = df['heuristics'].dropna()
    return heuristics_matrix
