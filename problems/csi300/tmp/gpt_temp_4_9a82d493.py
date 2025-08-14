def heuristics_v2(df):
    def alpha1(row):
        return row['high'] - row['low']
    
    def alpha2(row):
        return row['close'] - row['open']
    
    def alpha3(row):
        return (row['high'] - row['open']) / (row['high'] - row['low'])
    
    df['alpha1'] = df.apply(alpha1, axis=1)
    df['alpha2'] = df.apply(alpha2, axis=1)
    df['alpha3'] = df.apply(alpha3, axis=1)
    
    heuristics_matrix = pd.DataFrame(index=df.index)
    for i in range(1, 4):
        col_name = f'alpha{i}'
        heuristics_matrix[f'{col_name}_momentum_5'] = df[col_name].rolling(window=5).mean()
        heuristics_matrix[f'{col_name}_momentum_10'] = df[col_name].rolling(window=10).mean()
    
    return heuristics_matrix
