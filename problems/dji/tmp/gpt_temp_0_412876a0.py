def heuristics_v2(df):
    # Calculate Adjusted High-Low Spread
    df['high_low_spread'] = df['high'] - df['low']
    
    # Calculate Close-to-Open Return
    df['close_open_return'] = (df['close'] - df['open']) / df['open']
