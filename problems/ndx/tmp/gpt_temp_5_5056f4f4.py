def heuristics_v2(df):
    avg_price = (df['high'] + df['low']) / 2
    roc_period = 14
    roc_line = ((avg_price - avg_price.shift(roc_period)) / avg_price.shift(roc_period)) * 100
    weights = list(range(1, roc_period+1))
    heuristics_matrix = roc_line.rolling(window=roc_period).apply(lambda x: np.average(x, weights=weights), raw=True).dropna()
    
    return heuristics_matrix
