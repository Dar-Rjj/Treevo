def heuristics_v2(df):
    # Calculate daily returns
    daily_returns = df['close'].pct_change().fillna(0)
    # Calculate the standard deviation of daily volumes
    volume_std = df['volume'].std()
    # Adjust daily returns by the inverse of volume standard deviation to emphasize volume stability
    adjusted_returns = daily_returns / (volume_std + 1e-6)  # Add small value to avoid division by zero
    # Calculate the cumulative product of adjusted daily returns
    cumulative_product = (1 + adjusted_returns).cumprod()
    # Compute the heuristic factor
    heuristic_factor = cumulative_product[-1] - 1
    # Create a Series for the heuristic factor, repeating it for each date in the DataFrame
    heuristics_matrix = pd.Series([heuristic_factor]*len(df), index=df.index)
    return heuristics_matrix
