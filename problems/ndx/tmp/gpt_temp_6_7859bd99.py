def heuristics_v2(df):
    short_window = 12
    long_window = 26
    smoothed_factor = 5
    
    short_ema = df['close'].ewm(span=short_window, adjust=False).mean()
    long_ema = df['close'].ewm(span=long_window, adjust=False).mean()
    ratio = short_ema / long_ema
    weights = [0.4, 0.3, 0.2, 0.1]  # Custom weights for the smoothing
    heuristics_matrix = ratio.rolling(window=smoothed_factor).apply(lambda x: (x * weights).sum(), raw=True).dropna()
    
    return heuristics_matrix
