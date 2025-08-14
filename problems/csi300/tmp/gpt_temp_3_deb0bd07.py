def heuristics_v2(df):
    # Compute Daily Intraday Returns (Close - Open)
    df['intraday_return'] = df['close'] - df['open']
