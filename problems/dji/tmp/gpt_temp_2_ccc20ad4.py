def heuristics_v2(df):
    # Calculate Daily Returns
    df['daily_returns'] = df['close'].diff()
