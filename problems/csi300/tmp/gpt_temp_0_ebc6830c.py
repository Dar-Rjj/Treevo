def heuristics_v2(df):
    # Calculate Daily Price Change
    df['daily_price_change'] = df['close'] - df['close'].shift(1)
    
    # Calculate 10-Day and 20-Day Moving Averages of Daily Price Changes
    df['10_day_ma'] = df['daily_price_change'].rolling(window=10).mean()
    df['20_day_ma'] = df['daily_price_change'].rolling(window=20).mean()
    
    # Combine Moving Averages
    df['ma_diff'] = df['10_day_ma'] - df['20_day_ma']
    df['combined_ma'] = df['ma_diff'].rolling(window=5).mean()
    
    # Calculate Volume and Amount Weighted Daily Return
    df['daily_return'] = df['close'] / df['open']
