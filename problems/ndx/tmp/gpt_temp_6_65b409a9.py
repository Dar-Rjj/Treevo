def heuristics_v2(df):
    # Calculate daily log returns
    daily_log_returns = (df['close'] / df['close'].shift(1)).apply(np.log)
    # Calculate the standard deviation of the daily volumes
    volume_std = df['volume'].std()
    # Calculate the average of daily log returns
    avg_daily_log_returns = daily_log_returns.mean()
    # Compute the heuristic factor
    heuristic_factor = avg_daily_log_returns / volume_std if volume_std > 0 else 0
    # Create a Series for the heuristic factor, repeating it for each date in the DataFrame
    heuristics_matrix = pd.Series([heuristic_factor]*len(df), index=df.index)
    return heuristics_matrix
