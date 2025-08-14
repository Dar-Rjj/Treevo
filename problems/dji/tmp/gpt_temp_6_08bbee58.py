def heuristics_v2(df):
    step1 = df['close'].rolling(window=10).mean()
    step2 = df['close'].rolling(window=30).mean()
    step3 = (df['high'] / df['low']).rolling(window=5).apply(lambda x: np.prod(x), raw=True)
    heuristics_matrix = (step1 - step2) + step3
    return heuristics_matrix
