def heuristics_v2(df):
    # Calculate daily returns
    daily_returns = df['close'].pct_change()
    # Calculate the exponential moving average of daily returns with a span of 20 days
    ema_daily_returns = daily_returns.ewm(span=20, adjust=False).mean()
    # Calculate the standard deviation and sum of the last 10 days' volumes
    volume_std = df['volume'][-10:].std()
    close_sum = df['close'][-10:].sum()
    # Compute the volume-to-price ratio
    vol_to_price_ratio = volume_std / close_sum if close_sum > 0 else 0
    # Compute the heuristic factor
    heuristic_factor = ema_daily_returns.mean() / vol_to_price_ratio if vol_to_price_ratio > 0 else 0
    # Create a Series for the heuristic factor, repeating it for each date in the DataFrame
    heuristics_matrix = pd.Series([heuristic_factor]*len(df), index=df.index)
    return heuristics_matrix
