def heuristics_v2(df):
    step1 = (df['high'] - df['low']).abs().rolling(window=5).apply(lambda x: np.prod(x), raw=True)
    step2 = df['close'].rolling(window=5).sum()
    heuristics_matrix = step1 / step2
    return heuristics_matrix
