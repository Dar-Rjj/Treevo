def heuristics_v2(df):
    # Calculate daily high-low price difference
    daily_range = df['high'] - df['low']
    
    # Compute 20-day moving average of the daily range
    moving_avg_daily_range = daily_range.rolling(window=20).mean()
    
    # Return the computed heuristic as a pandas Series
    return heuristics_matrix
