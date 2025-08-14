def heuristics_v2(df, n=10):
    # Calculate Daily Price Movement (High - Low)
    daily_price_movement = df['high'] - df['low']
    
    # Calculate Price Gap (Open - Close)
    price_gap = df['open'] - df['close']
    
    # Calculate Raw Momentum (Sum of (High - Low) over n days)
    raw_momentum = daily_price_movement.rolling(window=n).sum()
    
    # Calculate Volume-Weighted Average Price (VWAP)
    vwap = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Incorporate Market Depth (Amount / Volume)
    market_depth = df['amount'] / df['volume']
    
    # Calculate Intraday Volatility (High - Low)
    intraday_volatility = daily_price_movement
    
    # Adjust Raw Momentum
    adjusted_momentum = (raw_momentum * (vwap + market_depth)) / intraday_volatility
    
    # Calculate Final Volume-Adjusted Momentum
    average_volume = df['volume'].rolling(window=n).mean()
    total_average_volume = df['volume'].mean()
    final_volume_adjusted_momentum = adjusted_momentum * (average_volume / total_average_volume)
    
    # Integrate Multi-Day Momentum
    five_day_return = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
