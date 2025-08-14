def heuristics_v2(df):
    daily_returns = df['close'].pct_change()
    avg_daily_returns = daily_returns.mean()
    volume_std = df['volume'].std()
    volume_mean = df['volume'].mean()
    if volume_mean > 0:
        volume_coefficient_of_variation = volume_std / volume_mean
    else:
        volume_coefficient_of_variation = 0
    heuristic_factor = avg_daily_returns * volume_coefficient_of_variation
    heuristics_matrix = pd.Series([heuristic_factor]*len(df), index=df.index)
    return heuristics_matrix
