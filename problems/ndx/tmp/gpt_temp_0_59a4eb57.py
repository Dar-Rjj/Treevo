def heuristics_v2(df):
    # Calculate short-term and long-term moving averages
    short_ma = df['close'].rolling(window=10).mean()
    long_ma = df['close'].rolling(window=50).mean()
    
    # Compute the logarithmic difference between the moving averages
    log_diff = (short_ma - long_ma).apply(np.log)
    
    # Calculate the RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Apply a custom heuristic to combine the logarithmic difference and RSI
    heuristics_matrix = (log_diff + rsi).rank(pct=True)
    
    return heuristics_matrix
