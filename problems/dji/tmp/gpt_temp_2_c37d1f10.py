def heuristics_v2(df):
    # Calculate Exponential Daily Returns
    df['daily_return'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
