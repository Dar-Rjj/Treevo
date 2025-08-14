def heuristics_v2(df):
    # Calculate Raw Returns
    df['returns'] = df['close'].pct_change()
