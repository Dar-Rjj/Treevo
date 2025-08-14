def heuristics_v2(df):
    # Calculate Daily Returns
    df['daily_return'] = df['close'].diff()
