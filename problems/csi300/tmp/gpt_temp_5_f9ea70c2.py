def heuristics_v2(df):
    # Calculate Daily Price Change
    df['daily_price_change'] = df['close'] - df['close'].shift(1)
    
    # Calculate Rolling Average of Daily Price Changes (10-Day Moving Average)
    df['rolling_avg_price_change'] = df['daily_price_change'].rolling(window=10).mean()
    
    # Calculate Volume and Amount Weighted Daily Return
    df['daily_return'] = df['close'] / df['open']
