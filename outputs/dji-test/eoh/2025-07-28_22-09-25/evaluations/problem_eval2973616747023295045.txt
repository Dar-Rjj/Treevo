def heuristics_v2(df):
    max_high = df['high'].rolling(window=20).max()
    adjusted_ratio = (df['close'] / max_high) * np.log(df['volume'])
    return heuristics_matrix
