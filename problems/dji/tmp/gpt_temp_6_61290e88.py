def heuristics_v2(df):
    # Calculate Daily Return
    df['daily_return'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
