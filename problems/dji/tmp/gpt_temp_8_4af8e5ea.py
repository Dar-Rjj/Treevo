def heuristics_v2(df):
    heuristics_matrix = (df['close'].pct_change() - (0.5 * df['open'].pct_change() + 0.5 * df['close'].pct_change())).shift(-1) * np.sqrt(df['volume'])
    heuristics_matrix = heuristics_matrix.dropna()
    return heuristics_matrix
