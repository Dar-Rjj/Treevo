def heuristics_v2(df):
    # Intraday High-to-Low Range
    daily_range = (df['high'] - df['low']) / df['open']
    
    # Open to Close Momentum
    open_to_close_return = (df['close'] - df['open']) / df['open']
