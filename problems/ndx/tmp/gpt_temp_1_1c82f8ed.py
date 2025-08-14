def heuristics_v2(df):
    # Calculate the percentage change for each column
    pct_change = df.pct_change()
    
    # Define weights for each feature; adjust as necessary based on empirical analysis
    weights = {'open': 0.2, 'high': 0.2, 'low': 0.2, 'close': 0.2, 'volume': 0.2}
    
    # Compute the weighted sum to form the heuristic score
    heuristics_matrix = (pct_change * pd.Series(weights)).sum(axis=1)
    
    return heuristics_matrix
