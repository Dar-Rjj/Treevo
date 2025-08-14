def heuristics_v2(df):
    # Calculate the 50-period Simple Moving Average (SMA) of the closing price
    sma_50 = df['close'].rolling(window=50).mean()
    
    # Calculate the ratio of the current closing price to the 50-period SMA
    close_to_sma_ratio = df['close'] / sma_50
    
    # Calculate the 5-day logarithmic return
    log_return_5 = np.log(df['close']).diff(5)
    
    # Combine the close-to-SMA ratio and the 5-day logarithmic return
    heuristics_matrix = (close_to_sma_ratio + log_return_5).rank(pct=True)
    
    return heuristics_matrix
