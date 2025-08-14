def heuristics_v2(df):
    def calc_heuristic(row):
        return (row['close'] - row['open']) / row['volume']
    
    heuristics_matrix = df.apply(calc_heuristic, axis=1)
    return heuristics_matrix
