def heuristics_v2(df, n_days=10):
    # Calculate N-Day Cumulative Return
    df['cumulative_return'] = (df['close'].pct_change().rolling(window=n_days).sum())
