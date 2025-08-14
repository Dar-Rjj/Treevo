def heuristics_v2(df):
    # Calculate short-term and long-term simple moving averages
    short_sma = df['close'].rolling(window=50).mean()
    long_sma = df['close'].rolling(window=100).mean()
    
    # Calculate the logarithmic return over a 20-day period
    log_return = np.log(df['close']).diff(20)
    
    # Compute the ratio between short and long simple moving averages
    sma_ratio = short_sma / long_sma
    
    # Apply a custom heuristic to combine the SMA ratio and logarithmic return
    heuristics_matrix = (sma_ratio + log_return).rank(pct=True)
    
    return heuristics_matrix
