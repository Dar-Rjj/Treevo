def heuristics_v2(df):
    # Calculate the 30-day moving average of daily high minus low prices
    diff_high_low = df['high'] - df['low']
    moving_avg_diff_high_low = diff_high_low.rolling(window=30).mean()
    
    # Calculate the 30-day moving average of the closing prices
    moving_avg_close = df['close'].rolling(window=30).mean()
    
    # Calculate the standard deviation of the 30-day moving average of the closing prices
    close_moving_avg_std = moving_avg_close.std()
    
    # Compute the heuristic factor
    heuristic_factor = (moving_avg_diff_high_low - moving_avg_close) / close_moving_avg_std if close_moving_avg_std > 0 else 0
    
    # Create a Series for the heuristic factor, ensuring it is indexed correctly
    heuristics_matrix = pd.Series(heuristic_factor, index=df.index)
    return heuristics_matrix
