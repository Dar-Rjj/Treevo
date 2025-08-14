def heuristics_v2(df):
    # Calculate Intraday Return
    df['intraday_return'] = df['close'] - df['open']
