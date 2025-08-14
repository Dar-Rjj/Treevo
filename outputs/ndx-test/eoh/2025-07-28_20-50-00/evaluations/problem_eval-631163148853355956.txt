def heuristics_v2(df):
    avg_daily_range = (df['high'] - df['low']).mean()
    avg_daily_return = df['close'].pct_change().mean()
    heuristic_factor = avg_daily_range / avg_daily_return if avg_daily_return != 0 else 0
    heuristics_matrix = pd.Series([heuristic_factor]*len(df), index=df.index)
    return heuristics_matrix
