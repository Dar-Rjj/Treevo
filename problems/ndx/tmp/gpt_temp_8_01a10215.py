def heuristics_v2(df):
    # Calculate daily returns
    daily_returns = df['close'].pct_change()
    # Calculate the 50th and 90th percentiles of the daily volumes
    volume_50th = df['volume'].quantile(0.50)
    volume_90th = df['volume'].quantile(0.90)
    # Ensure the 90th percentile volume is not zero to avoid division by zero
    if volume_90th == 0:
        return pd.Series([0]*len(df), index=df.index)
    # Ratio of 50th to 90th percentile of volume
    volume_ratio = volume_50th / volume_90th
    # Calculate the mean of daily returns
    mean_daily_returns = daily_returns.mean()
    # Compute the heuristic factor
    heuristic_factor = mean_daily_returns * (1 + volume_ratio) if volume_90th > 0 else 0
    # Create a Series for the heuristic factor, repeating it for each date in the DataFrame
    heuristics_matrix = pd.Series([heuristic_factor]*len(df), index=df.index)
    return heuristics_matrix
