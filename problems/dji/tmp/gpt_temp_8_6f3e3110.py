def heuristics_v2(df):
    high_low_ratio = df['high'] / df['low']
    weights = pd.Series(range(1,21))
    heuristics_matrix = high_low_ratio.rolling(window=20).apply(lambda x: (x*weights).sum() / weights.sum(), raw=True)
    return heuristics_matrix
