def heuristics_v2(df):
    # Calculate Daily Price Change
    df['price_change'] = df['close'] - df['close'].shift(1)
    
    # Calculate Rolling Average of Daily Price Changes (15-Day Moving Average)
    df['rolling_avg_price_change'] = df['price_change'].rolling(window=15).mean()
    
    # Calculate Volume and Amount Weighted Daily Return
    df['daily_return'] = df['close'] / df['open']
