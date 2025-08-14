def heuristics_v2(df):
    # Calculate the 30-day moving averages of daily high and low prices
    moving_avg_high = df['high'].rolling(window=30).mean()
    moving_avg_low = df['low'].rolling(window=30).mean()
    
    # Calculate the 30-day moving average of daily closing prices
    moving_avg_close = df['close'].rolling(window=30).mean()
    
    # Compute the heuristic factor
    heuristic_factor = (moving_avg_high / moving_avg_low) - moving_avg_close
    
    # Create a Series for the heuristic factor, ensuring it is indexed correctly
    heuristics_matrix = pd.Series(heuristic_factor, index=df.index)
    return heuristics_matrix
