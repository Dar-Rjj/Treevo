def heuristics_v2(df):
    # Calculate Intraday Range
    df['intraday_range'] = df['high'] - df['low']
    
    # Calculate Open-to-Close Return
    df['open_to_close_return'] = (df['close'].shift(1) - df['open']) / df['open']
