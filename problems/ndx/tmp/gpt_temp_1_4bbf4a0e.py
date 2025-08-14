def heuristics_v2(df):
    # Calculate the 30-day moving average of daily high and low prices
    avg_high_low = (df['high'] + df['low']) / 2
    moving_avg_high_low = avg_high_low.rolling(window=30).mean()
    
    # Calculate the 30-day moving average of the closing prices
    moving_avg_close = df['close'].rolling(window=30).mean()
    
    # Calculate the daily high and low price differences
    high_low_diff = df['high'] - df['low']
    # Calculate the standard deviation of the daily high and low price differences over a 30-day period
    high_low_diff_std = high_low_diff.rolling(window=30).std()
    
    # Compute the heuristic factor
    heuristic_factor = (moving_avg_high_low - moving_avg_close) / high_low_diff_std
    heuristic_factor = heuristic_factor.fillna(0)
    
    # Create a Series for the heuristic factor, ensuring it is indexed correctly
    heuristics_matrix = pd.Series(heuristic_factor, index=df.index)
    return heuristics_matrix
