def heuristics_v2(df):
    short_window = 10
    long_window = 50
    smooth_window = 7
    
    log_short_ma = df['close'].rolling(window=short_window).mean().apply(np.log)
    log_long_ma = df['close'].rolling(window=long_window).mean().apply(np.log)
    diff = log_short_ma - log_long_ma
    heuristics_matrix = diff.rolling(window=smooth_window).mean().dropna()
    
    return heuristics_matrix
