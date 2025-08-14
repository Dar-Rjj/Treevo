def heuristics_v2(df):
    # Calculate the daily trading range
    daily_range = df['high'] - df['low']
    # Calculate the mean of the daily trading range
    range_mean = daily_range.mean()
    # Calculate the mean of the daily closing prices
    close_mean = df['close'].mean()
    # Compute the heuristic factor
    heuristic_factor = range_mean / close_mean if close_mean > 0 else 0
    # Create a Series for the heuristic factor, repeating it for each date in the DataFrame
    heuristics_matrix = pd.Series([heuristic_factor]*len(df), index=df.index)
    return heuristics_matrix
