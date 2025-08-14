def heuristics_v2(df):
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
