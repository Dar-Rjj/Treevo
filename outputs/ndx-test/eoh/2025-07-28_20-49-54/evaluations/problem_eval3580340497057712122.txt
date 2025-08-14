def heuristics_v2(df):
    # Calculate short-term and long-term exponential moving averages
    short_ema = df['close'].ewm(span=12, adjust=False).mean()
    long_ema = df['close'].ewm(span=26, adjust=False).mean()
    
    # Calculate the logarithmic price-volume relationship
    pv_relationship = np.log(df['close'] * df['volume'])
    
    # Compute the ratio of short and long exponential moving averages
    ema_ratio = short_ema / long_ema
    
    # Apply a custom heuristic to combine the exponential moving average ratio and logarithmic price-volume relationship
    heuristics_matrix = (ema_ratio + pv_relationship).rank(pct=True)
    
    return heuristics_matrix
