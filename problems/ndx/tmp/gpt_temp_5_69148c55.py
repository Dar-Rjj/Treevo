def heuristics_v2(df):
    short_window = 15
    long_window = 60
    smoothed_factor = 0.2
    
    short_wma = df['close'].rolling(window=short_window).apply(lambda x: (x * pd.Series(range(1, len(x) + 1))).sum() / (pd.Series(range(1, len(x) + 1))).sum(), raw=False)
    long_wma = df['close'].rolling(window=long_window).apply(lambda x: (x * pd.Series(range(1, len(x) + 1))).sum() / (pd.Series(range(1, len(x) + 1))).sum(), raw=False)
    
    short_smoothed = short_wma.ewm(alpha=smoothed_factor).mean()
    long_smoothed = long_wma.ewm(alpha=smoothed_factor).mean()
    
    heuristics_matrix = (short_smoothed - long_smoothed).dropna()
    
    return heuristics_matrix
