def heuristics_v2(df):
    short_window = 10
    long_window = 50
    smoothing_span = 10
    
    short_ma = df['close'].rolling(window=short_window).mean()
    long_ma = df['close'].rolling(window=long_window).mean()
    log_diff = np.log(short_ma) - np.log(long_ma)
    heuristics_matrix = log_diff.ewm(span=smoothing_span, adjust=False).mean().dropna()
    
    return heuristics_matrix
