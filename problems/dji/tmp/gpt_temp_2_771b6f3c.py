def heuristics_v2(df):
    def calc_heuristic(row):
        if row.name < 20:
            return 0
        avg_close = df['close'].iloc[row.name-20:row.name].mean()
        return row['close'] / avg_close - 1
    
    heuristics_matrix = df.apply(calc_heuristic, axis=1)
    return heuristics_matrix
