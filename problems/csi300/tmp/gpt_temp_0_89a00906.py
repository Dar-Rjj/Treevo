def heuristics_v2(df):
    # Calculate Daily Close-to-Close Return
    df['daily_return'] = df['close'].pct_change()
