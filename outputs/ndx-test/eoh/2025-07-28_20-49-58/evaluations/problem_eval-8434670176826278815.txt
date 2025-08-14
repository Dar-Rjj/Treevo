def heuristics_v2(df):
    # Calculate the 30-day weighted average of closing prices, with more recent prices having a higher weight
    close_wavg_30 = df['close'].rolling(window=30).apply(lambda x: (x * (np.arange(1, len(x) + 1) / np.sum(np.arange(1, len(x) + 1)))).sum(), raw=False)
    
    # Compute the alpha factor as the ratio of the current closing price to the 30-day weighted average of closing prices
    heuristics_matrix = (df['close'] / close_wavg_30).dropna()
    
    return heuristics_matrix
