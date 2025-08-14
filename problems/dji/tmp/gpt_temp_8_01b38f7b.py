def heuristics_v4(df):
    weighted_diff = (df['high'] - df['low']) * df['volume']
    heuristics_matrix = weighted_diff.rolling(window=5).mean()
    return heuristics_matrix
