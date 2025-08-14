def heuristics_v2(df):
    # Compute short-term log return
    short_term_log_return = (df['close'] / df['close'].shift(7)).apply(np.log)
    
    # Compute medium-term log return
    medium_term_log_return = (df['close'] / df['close'].shift(30)).apply(np.log)
    
    # Compute long-term log return
    long_term_log_return = (df['close'] / df['close'].shift(90)).apply(np.log)
    
    # Combine the log returns into a heuristic matrix
    heuristics_matrix = short_term_log_return + 2 * medium_term_log_return + 3 * long_term_log_return
    
    return heuristics_matrix
