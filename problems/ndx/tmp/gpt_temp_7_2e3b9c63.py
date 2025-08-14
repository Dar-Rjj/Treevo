def heuristics_v2(df):
    # Calculate 20-day Price Momentum
    df['price_momentum'] = df['close'] - df['close'].shift(20)
    
    # Identify Breakout Days
    df['high_low_range'] = df['high'] - df['low']
    avg_high_low_range = df['high_low_range'].rolling(window=20).mean()
    df['is_breakout'] = (df['high_low_range'] > 2 * avg_high_low_range).astype(int)
    
    # Volume-Adjusted Breakout Impact
    df['daily_return'] = (df['close'] / df['open']) - 1
