def heuristics_v2(df):
    # Calculate Intraday Range
    intraday_range = df['high'] - df['low']
    
    # Calculate Close Position in Range
    close_position_in_range = (df['close'] - df['low']) / intraday_range
    
    # Calculate Daily Return
    daily_return = (df['close'].pct_change()).fillna(0)
