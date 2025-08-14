def heuristics_v2(df):
    # Calculate Intraday High-Low Spread
    df['intraday_high_low_spread'] = df['high'] - df['low']
    
    # Calculate Intraday Return
    df['intraday_return'] = (df['close'] - df['open']) / df['open']
