def heuristics_v2(df):
    # Calculate the volume weighted average price (VWAP)
    vwap = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Calculate the high-to-low price ratio and take the log
    hl_ratio_log = np.log(df['high'] / df['low'])
    
    # Compute the absolute difference between closing price and VWAP
    cp_vwap_diff_abs = abs(df['close'] - vwap)
    
    # Apply a custom heuristic to combine the logarithmic high-to-low price ratio and the absolute difference
    heuristics_matrix = (hl_ratio_log + cp_vwap_diff_abs).rank(pct=True)
    
    return heuristics_matrix
