def heuristics_v2(df):
    # Calculate daily range
    daily_range = df['high'] - df['low']
    # Calculate the standard deviation of daily closing prices
    close_std = df['close'].std()
    # Calculate the average of daily range
    avg_daily_range = daily_range.mean()
    # Compute the heuristic factor
    heuristic_factor = avg_daily_range / close_std if close_std > 0 else 0
    # Create a Series for the heuristic factor, repeating it for each date in the DataFrame
    heuristics_matrix = pd.Series([heuristic_factor]*len(df), index=df.index)
    return heuristics_matrix
