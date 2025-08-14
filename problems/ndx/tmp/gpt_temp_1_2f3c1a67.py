def heuristics_v2(df):
    short_window = 10
    long_window = 50
    smoothed_factor = 0.3
    
    log_close = np.log(df['close'])
    
    short_ma = log_close.rolling(window=short_window).mean()
    long_ma = log_close.rolling(window=long_window).mean()
    ratio = short_ma / long_ma
    heuristics_matrix = ratio.ewm(alpha=smoothed_factor).mean().dropna()
    
    return heuristics_matrix
