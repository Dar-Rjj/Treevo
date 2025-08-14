def heuristics_v2(df):
    weights = np.exp(np.linspace(0, -1, 50))
    df['wma'] = (df['close'].rolling(window=50).apply(lambda x: np.sum(weights * x) / np.sum(weights), raw=False))
    heuristics_matrix = df['wma'] - df['low'].rolling(window=50).min()
    return heuristics_matrix
