def heuristics_v2(df):
    # Calculate High-Low Spread
    df['high_low_spread'] = df['high'] - df['low']
    
    # Compute 20-Day Moving Average of High-Low Spread
    df['high_low_spread_ma_20'] = df['high_low_spread'].rolling(window=20).mean()
    
    # Calculate Cumulative Return Over 20 Days
    df['daily_return'] = df['close'].pct_change()
