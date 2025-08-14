def heuristics_v2(df):
    # Calculate High-Low Range
    df['high_low_range'] = df['high'] - df['low']
    
    # Adjust for Opening Gap
    df['prev_close'] = df['close'].shift(1)
    df['open_gap'] = df['open'] - df['prev_close']
    
    # Calculate Price Movement Ratio
    df['price_movement_ratio'] = (df['close'] - df['open']) / df['open']
    
    # Incorporate Volume Momentum
    df['volume_change'] = df['volume'] - df['volume'].shift(1)
    df['adjusted_range'] = df['high_low_range'] * df['volume_change']
    
    # Apply Time Series Momentum
    lookback_period = 10
    df['avg_adjusted_range'] = df['adjusted_range'].rolling(window=lookback_period).mean()
    df['ts_momentum_diff'] = df['adjusted_range'] - df['avg_adjusted_range']
    
    # Calculate 5-Day Rolling Momentum
    df['rolling_price_ratio'] = df['price_movement_ratio'].rolling(window=5).sum()
    df['rolling_high_low_range'] = df['high_low_range'].rolling(window=5).sum()
    df['rolling_momentum'] = df['rolling_price_ratio'] / df['rolling_high_low_range']
    
    # Adjust Momentum for Volume
    df['volume_adjusted_momentum'] = df['rolling_momentum'] * df['volume']
    
    # Integrate Volume-Adjusted Momentum and Volatility into Final Factor
    df['daily_volatility'] = (df['high'] - df['low']).abs()
    df['close_open_return'] = (df['close'] - df['open']) / df['open']
