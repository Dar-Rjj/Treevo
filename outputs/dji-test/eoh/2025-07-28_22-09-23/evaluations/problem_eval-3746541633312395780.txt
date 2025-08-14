def heuristics_v2(df):
    def roc(series, n=14):
        return (series / series.shift(n) - 1) * 100
    
    def adl_modified(df):
        money_flow_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        money_flow_volume = money_flow_multiplier * df['volume']
        return money_flow_volume.rolling(window=14).sum()  # Using a rolling sum instead of cumsum for a more dynamic ADL
    
    def wma_roc(roc_series, n=14):
        weights = np.arange(1, n+1)
        return roc_series.rolling(window=n).apply(lambda x: np.sum(weights * x) / weights.sum(), raw=True)
    
    close_roc = roc(df['close'])
    adl_line = adl_modified(df)
    wma_close_roc = wma_roc(close_roc)
    heuristics_matrix = (wma_close_roc + adl_line) / 2
    return heuristics_matrix
