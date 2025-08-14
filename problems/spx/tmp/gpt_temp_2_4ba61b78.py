def heuristics_v2(df):
    # Calculate Daily Price Change
    df['daily_price_change'] = df['close'] - df['close'].shift(1)
    
    # Calculate Intraday Return
    df['intraday_return'] = (df['close'] / df['open']) - 1
