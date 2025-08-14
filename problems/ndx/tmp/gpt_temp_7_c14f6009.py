def heuristics_v2(df):
    # Calculate the volume weighted average price (VWAP)
    vwap = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Calculate the logarithmic return
    log_return = df['close'].pct_change().apply(lambda x: np.log(1 + x))
    
    # Compute the ratio of closing price to VWAP
    close_to_vwap_ratio = df['close'] / vwap
    
    # Apply a custom heuristic to combine the logarithmic return and the close-to-VWAP ratio
    heuristics_matrix = (log_return + close_to_vwap_ratio).rank(pct=True)
    
    return heuristics_matrix
