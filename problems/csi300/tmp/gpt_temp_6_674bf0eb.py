def heuristics_v2(df):
    # Calculate Daily Price Change
    df['daily_price_change'] = df['close'] - df['close'].shift(1)
    
    # Calculate Rolling Average of Daily Price Changes
    df['price_change_10_day_ma'] = df['daily_price_change'].rolling(window=10).mean()
    
    # Calculate Volume and Amount Weighted Daily Return
    df['daily_return'] = df['close'] / df['open']
