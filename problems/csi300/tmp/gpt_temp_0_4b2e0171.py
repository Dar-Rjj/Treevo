def heuristics_v2(df):
    # Calculate Daily Midpoint
    df['midpoint'] = (df['high'] + df['low']) / 2
    
    # Multi-Day Average Midpoint (5-day window)
    df['avg_midpoint'] = df['midpoint'].rolling(window=5).mean()
    
    # Compute Volume Adjusted Momentum
    df['volume_adjusted_momentum'] = (df['close'] - df['close'].shift(1)) * df['volume']
    
    # Intraday High-Low Range
    df['intraday_range'] = df['high'] - df['low']
    
    # Daily Returns
    df['daily_returns'] = df['close'] - df['open']
