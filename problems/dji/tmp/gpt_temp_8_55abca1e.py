def heuristics_v2(df):
    # Calculate Daily Return
    df['daily_return'] = df['close'].pct_change()
