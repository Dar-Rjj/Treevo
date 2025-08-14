def heuristics_v2(df):
    # Calculate the 20-day simple moving average (SMA)
    sma_20 = df['close'].rolling(window=20).mean()
    
    # Calculate the high-to-20-day-SMA ratio
    high_sma_ratio = df['high'] / sma_20
    
    # Compute the logarithmic difference between the closing price and the 10-day SMA
    sma_10 = df['close'].rolling(window=10).mean()
    log_diff_close_sma_10 = (df['close'] - sma_10).apply(lambda x: np.log(x + 1))
    
    # Apply a custom heuristic to combine the high-to-20-day-SMA ratio and the logarithmic difference
    heuristics_matrix = (high_sma_ratio + log_diff_close_sma_10).rank(pct=True)
    
    return heuristics_matrix
