def heuristics_v2(df):
    def heuristic1(row):
        return (row['close'] - row['open']) / (row['high'] - row['low'])
    
    def heuristic2(row):
        return (row['close'] - row['low']) / (row['high'] - row['low'])
    
    def heuristic3(row):
        return (row['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    
    heuristics_matrix = pd.DataFrame(index=df.index)
    heuristics_matrix['h1'] = df.apply(heuristic1, axis=1)
    heuristics_matrix['h2'] = df.apply(heuristic2, axis=1)
    heuristics_matrix['h3'] = df.apply(heuristic3, axis=1)
    heuristics_matrix = heuristics_matrix.fillna(0)
    return heuristics_matrix
