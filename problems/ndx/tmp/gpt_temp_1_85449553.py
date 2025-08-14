def heuristics_v2(df):
    # Calculate the 30-day moving average of daily high and low prices
    moving_avg_high = df['high'].rolling(window=30).mean()
    moving_avg_low = df['low'].rolling(window=30).mean()
    
    # Sum of the moving averages
    sum_moving_avg_high = moving_avg_high.sum()
    sum_moving_avg_low = moving_avg_low.sum()
    
    # Calculate the 50-day simple moving average of the closing prices
    sma_close_50 = df['close'].rolling(window=50).mean()
    
    # Compute the heuristic factor
    heuristic_factor = (sum_moving_avg_high / sum_moving_avg_low) - sma_close_50
    
    # Create a Series for the heuristic factor, ensuring it is indexed correctly
    heuristics_matrix = pd.Series(heuristic_factor, index=df.index)
    return heuristics_matrix
