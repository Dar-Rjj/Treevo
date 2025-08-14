def heuristics_v2(df):
    # Calculate daily returns
    daily_returns = df['close'].pct_change()
    # Calculate the exponential moving average of daily returns with a span of 20 days
    ema_daily_returns = daily_returns.ewm(span=20, adjust=False).mean()
    # Calculate the IQR of the daily volumes
    volume_iqr = df['volume'].quantile(0.75) - df['volume'].quantile(0.25)
    # Compute the heuristic factor
    heuristic_factor = ema_daily_returns.mean() / volume_iqr if volume_iqr > 0 else 0
    # Create a Series for the heuristic factor, repeating it for each date in the DataFrame
    heuristics_matrix = pd.Series([heuristic_factor]*len(df), index=df.index)
    return heuristics_matrix
