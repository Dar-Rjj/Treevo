def heuristics_v2(df):
    def compute_rate_of_change(df, window=10):
        return df['close'].pct_change(window)
    
    def compute_average_true_range(df, window=14):
        tr = df[['high', 'low', 'close']].apply(lambda x: max(x) - min(x), axis=1)
        atr = tr.rolling(window=window).mean()
        return atr
    
    def compute_log_volume_change(df, window=5):
        return np.log(df['volume']).diff(window)
    
    roc = compute_rate_of_change(df)
    atr = compute_average_true_range(df)
    lvc = compute_log_volume_change(df)
    
    heuristics_matrix = (roc / atr) + lvc
    return heuristics_matrix
