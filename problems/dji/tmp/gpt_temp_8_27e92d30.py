def heuristics_v2(df):
    step1 = (df['high'] / df['low']) * np.log(df['volume'])
    heuristics_matrix = step1.rolling(window=10).median()
    return heuristics_matrix
