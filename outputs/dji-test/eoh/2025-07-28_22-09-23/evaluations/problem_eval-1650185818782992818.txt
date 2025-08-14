def heuristics_v2(df):
    def calculate_heuristic(row):
        return (row['high'] - row['low']) / row['open']
    
    df['heuristic'] = df.apply(calculate_heuristic, axis=1)
    heuristics_matrix = df['heuristic'].rolling(window=30).mean()
    heuristics_matrix = heuristics_matrix + np.log(df['volume'])
    return heuristics_matrix
