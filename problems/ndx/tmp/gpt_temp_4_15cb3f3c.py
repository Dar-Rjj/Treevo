def heuristics_v2(df):
    # Calculate High-Low Range
    df['high_low_range'] = df['high'] - df['low']
    
    # Calculate 21-day Price Momentum
    df['price_momentum_21'] = df['close'] - df['close'].shift(21)
    
    # Identify Breakout Days
    df['avg_high_low_range_21'] = df['high_low_range'].rolling(window=21).mean()
    df['breakout'] = (df['high_low_range'] > 3 * df['avg_high_low_range_21']).astype(int)
    
    # Calculate Volume-Adjusted Breakout Impact
    df['daily_return'] = df['close'] - df['open']
