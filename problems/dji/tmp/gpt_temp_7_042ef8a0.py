def heuristics_v2(df):
    # Calculate Intraday Range
    intraday_range = df['high'] - df['low']
    
    # Calculate Close Position in Range
    close_position_in_range = (df['close'] - df['low']) / intraday_range
    
    # Calculate Close-to-Open Return
    close_to_open_return = (df['close'] - df['open']) / df['open']
