def heuristics_v2(df):
    # Calculate 20-day Price Momentum
    df['20_day_momentum'] = df['close'].diff(20)
    
    # Identify Breakout Days
    df['high_low_range'] = df['high'] - df['low']
    avg_range_20 = df['high_low_range'].rolling(window=20).mean()
    df['is_breakout'] = (df['high_low_range'] > 2 * avg_range_20).astype(int)
    
    # Calculate Volume-Adjusted Breakout Impact
    df['daily_return'] = df['close'] / df['open'] - 1
