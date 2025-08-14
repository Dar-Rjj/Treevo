def heuristics_v2(df):
    # Calculate daily log returns
    daily_log_returns = np.log(df['close']).diff()
    # Calculate the difference between the average and median of daily log returns
    diff_avg_median_returns = daily_log_returns.mean() - daily_log_returns.median()
    # Calculate the interquartile range of daily closing prices
    iqr_close = df['close'].quantile(0.75) - df['close'].quantile(0.25)
    # Compute the heuristic factor
    heuristic_factor = diff_avg_median_returns / iqr_close if iqr_close > 0 else 0
    # Create a Series for the heuristic factor, repeating it for each date in the DataFrame
    heuristics_matrix = pd.Series([heuristic_factor]*len(df), index=df.index)
    return heuristics_matrix
