def heuristics_v2(df):
    # Calculate Intraday Momentum
    df['high_low_diff'] = df['high'] - df['low']
    df['open_close_return'] = df['close'] / df['open'] - 1
