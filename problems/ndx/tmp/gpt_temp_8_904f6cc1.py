def heuristics_v2(df):
    abs_diff = (df['close'] - df['open']).abs() * df['volume'].apply(lambda x: max(1, x)).apply(np.log)
    heuristics_matrix = abs_diff.rolling(window=5).sum()
    return heuristics_matrix
