def heuristics(df):
    # Calculate Intraday Range
    df['intraday_range'] = df['high'] - df['low']
    
    # Calculate Close to Midpoint Deviation
    df['close_to_midpoint_deviation'] = df['close'] - (df['high'] + df['low']) / 2
    
    # Calculate Adjusted Intraday Reversal
    df['intraday_reversal'] = 2 * (df['high'] - df['low']) / (df['close'] + df['open'])
    df['adjusted_intraday_reversal'] = df['intraday_reversal'] * (1 + (df['close'] - df['close'].shift(10)) / df['close'].shift(10))
    
    # Calculate Previous Day Return
    df['previous_day_return'] = (df['close'].shift(1) - df['open']).fillna(0)
