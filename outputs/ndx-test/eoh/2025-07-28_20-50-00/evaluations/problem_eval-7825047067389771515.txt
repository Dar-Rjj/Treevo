def heuristics_v2(df):
    # Calculate daily returns
    daily_returns = df['close'].pct_change()
    # Calculate the weighted average of daily returns using volume as weight
    weighted_avg_daily_returns = (daily_returns * df['volume']).sum() / df['volume'].sum()
    # Calculate the standard deviation of the daily close prices
    close_std = df['close'].std()
    # Compute the heuristic factor
    heuristic_factor = weighted_avg_daily_returns / close_std if close_std > 0 else 0
    # Create a Series for the heuristic factor, repeating it for each date in the DataFrame
    heuristics_matrix = pd.Series([heuristic_factor] * len(df), index=df.index)
    return heuristics_matrix
