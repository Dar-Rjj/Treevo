def heuristics_v2(df):
    heuristics_matrix = (df['close'].pct_change()**2 + np.log(df['volume'])).dropna()
    return heuristics_matrix
