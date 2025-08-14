def heuristics_v2(df):
    # Calculate the 20-day moving average of daily high prices
    moving_avg_high = df['high'].rolling(window=20).mean()
    
    # Calculate the 10-day moving average of daily low prices
    moving_avg_low = df['low'].rolling(window=10).mean()
    
    # Calculate the median absolute deviation of daily closing prices
    close_mad = df['close'].mad()
    
    # Compute the heuristic factor
    heuristic_factor = (moving_avg_high - moving_avg_low) / close_mad if close_mad > 0 else 0
    
    # Create a Series for the heuristic factor, ensuring it is indexed correctly
    heuristics_matrix = pd.Series(heuristic_factor, index=df.index)
    return heuristics_matrix
