def heuristics_v2(df):
    # Calculate 5-day moving average of daily returns
    daily_returns = df['close'].pct_change()
    ma_5_daily_returns = daily_returns.rolling(window=5).mean()
    # Calculate 10-day moving average of daily volumes
    ma_10_volume = df['volume'].rolling(window=10).mean()
    # Compute the heuristic factor
    heuristic_factor = (ma_5_daily_returns / ma_10_volume).fillna(0)
    # Convert the Series to match the length of input DataFrame
    heuristics_matrix = pd.Series(heuristic_factor, index=df.index)
    return heuristics_matrix
