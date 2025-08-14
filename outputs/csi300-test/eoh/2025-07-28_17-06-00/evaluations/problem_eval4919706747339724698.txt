def heuristics_v2(df):
    def dynamic_window_size(volatility, base_window=10):
        return int(base_window * (1 + 2 * volatility))
    
    volatility = df['close'].pct_change().rolling(window=5).std()
    window_size = volatility.apply(lambda x: dynamic_window_size(x)).fillna(10)
    
    heuristics_matrix = (df['close'] / df['volume']).rolling(window=window_size, min_periods=1).mean()
    return heuristics_matrix
