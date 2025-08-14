def heuristics(df):
    # Calculate Intraday Return
    df['intraday_return'] = (df['high'] - df['low']) / df['close']
