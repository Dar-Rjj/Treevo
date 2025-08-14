def heuristics_v2(df):
    # Calculate the 30-day moving average of the absolute difference between daily high and low prices
    avg_diff_high_low = (df['high'] - df['low']).abs().rolling(window=30).mean()
    
    # Calculate the 30-day moving standard deviation of daily closing prices
    close_mstd = df['close'].rolling(window=30).std()
    
    # Compute the heuristic factor
    heuristic_factor = avg_diff_high_low / close_mstd
    
    # Replace NaN values with 0 to avoid division by zero issues
    heuristic_factor = heuristic_factor.fillna(0)
    
    # Create a Series for the heuristic factor, ensuring it is indexed correctly
    heuristics_matrix = pd.Series(heuristic_factor, index=df.index)
    return heuristics_matrix
