def heuristics_v2(df):
    heuristics_matrix = (df['close'].pct_change().shift(-1) / np.sqrt(df['volume'])).dropna()
    return heuristics_matrix
