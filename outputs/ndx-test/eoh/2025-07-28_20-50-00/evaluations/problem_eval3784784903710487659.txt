def heuristics_v2(df):
    daily_returns = df['close'].pct_change()
    ema_daily_returns = daily_returns.ewm(span=30, adjust=False).mean()
    volume_mean = df['volume'][-15:].mean()
    volume_std = df['volume'][-15:].std()
    volume_cv = volume_std / volume_mean if volume_mean > 0 else 0
    heuristic_factor = ema_daily_returns.mean() / volume_cv if volume_cv > 0 else 0
    heuristics_matrix = pd.Series([heuristic_factor]*len(df), index=df.index)
    return heuristics_matrix
