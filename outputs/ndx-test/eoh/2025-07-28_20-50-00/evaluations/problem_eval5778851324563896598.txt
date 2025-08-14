def heuristics_v2(df):
    def calculate_roc(series, n):
        return (series / series.shift(n) - 1) * 100
    
    def calculate_wma(series, weights):
        return (series * weights).sum() / weights.sum()
    
    roc = calculate_roc(df['close'] - df['open'], 10)
    wma_volume = df['volume'].rolling(window=10).apply(lambda x: calculate_wma(x, np.arange(1, len(x)+1)), raw=False)
    
    heuristics_matrix = roc + wma_volume
    return heuristics_matrix
