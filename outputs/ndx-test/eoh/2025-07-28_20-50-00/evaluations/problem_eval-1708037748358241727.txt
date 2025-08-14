def heuristics_v2(df):
    # Calculate daily returns
    daily_returns = df['close'].pct_change()
    # Sum of positive daily returns
    sum_positive_returns = daily_returns[daily_returns > 0].sum()
    # Calculate the absolute difference between consecutive daily returns
    abs_return_diffs = daily_returns.diff().abs().sum()
    # Compute the heuristic factor
    heuristic_factor = sum_positive_returns / abs_return_diffs if abs_return_diffs > 0 else 0
    # Create a Series for the heuristic factor, repeating it for each date in the DataFrame
    heuristics_matrix = pd.Series([heuristic_factor]*len(df), index=df.index)
    return heuristics_matrix
