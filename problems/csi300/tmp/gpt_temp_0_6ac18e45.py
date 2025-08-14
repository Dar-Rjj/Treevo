def heuristics_v2(df):
    # Calculate Intraday Return
    df['intraday_range'] = df['high'] - df['low']
    df['intraday_return'] = df['intraday_range'] / df['close']
