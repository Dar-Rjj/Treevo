def heuristics_v2(df):
    # Calculate Intraday Return
    df['intraday_return'] = (df['high'] - df['low']) / df['open']
