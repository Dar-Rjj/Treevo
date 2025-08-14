def heuristics_v2(df):
    def compute_factor(row):
        return (row['close'] - row['open']) / row['amount']
    
    df['factor'] = df.apply(compute_factor, axis=1)
    heuristics_matrix = df['factor'].rolling(window=5).mean()
    return heuristics_matrix
