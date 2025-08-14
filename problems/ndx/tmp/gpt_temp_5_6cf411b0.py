def heuristics_v2(df):
    daily_returns = df['close'].pct_change().dropna()
    volume_std = df['volume'].std()
    volume_mean = df['volume'].mean()
    if volume_mean == 0:
        return pd.Series([0]*len(df), index=df.index)
    volume_cov = volume_std / volume_mean
    if volume_cov > 0:
        weight = 1 / (volume_cov * len(daily_returns))
        weighted_daily_returns = daily_returns * weight
        heuristic_factor = weighted_daily_returns.sum()
    else:
        heuristic_factor = 0
    heuristics_matrix = pd.Series([heuristic_factor]*len(df), index=df.index)
    return heuristics_matrix
