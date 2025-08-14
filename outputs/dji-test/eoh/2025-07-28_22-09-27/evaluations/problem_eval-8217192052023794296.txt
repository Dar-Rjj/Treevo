def heuristics_v2(df):
    volume_price_ratio = df['volume'] / df['close']
    weights = np.arange(1, 11)
    heuristics_matrix = volume_price_ratio.rolling(window=10).apply(lambda x: np.sum(weights * x) / np.sum(weights), raw=True)
    return heuristics_matrix
