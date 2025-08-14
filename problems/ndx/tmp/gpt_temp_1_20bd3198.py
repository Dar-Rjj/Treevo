def heuristics_v2(df):
    short_window = 10
    long_window = 30
    short_ema = df['close'].ewm(span=short_window, adjust=False).mean()
    long_ema = df['close'].ewm(span=long_window, adjust=False).mean()
    ratio = short_ema / long_ema
    weights = np.array([1, 2, 3, 4, 5])
    heuristics_matrix = ratio.rolling(window=weights.size).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True).dropna()
    
    return heuristics_matrix
