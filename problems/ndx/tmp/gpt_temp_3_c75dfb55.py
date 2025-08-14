def heuristics_v2(df):
    # Calculate the 30-day moving sum of daily high and low prices
    moving_sum_high = df['high'].rolling(window=30).sum()
    moving_sum_low = df['low'].rolling(window=30).sum()
    
    # Compute the ratio of the 30-day moving sums
    ratio_high_low = moving_sum_high / moving_sum_low
    
    # Calculate the 50-day moving average of daily closing prices
    moving_avg_close_50 = df['close'].rolling(window=50).mean()
    
    # Compute the heuristic factor
    heuristic_factor = ratio_high_low - moving_avg_close_50
    
    # Create a Series for the heuristic factor, ensuring it is indexed correctly
    heuristics_matrix = pd.Series(heuristic_factor, index=df.index)
    return heuristics_matrix
