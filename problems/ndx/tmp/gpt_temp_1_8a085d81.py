def heuristics_v2(df):
    # Calculate Volume-Weighted Price
    df['volume_weighted_price'] = df['close'] * df['volume']
    
    # Calculate Simple Moving Average (SMA) of Volume-Weighted Prices
    sma_window = 20
    df['sma_volume_weighted_price'] = df['volume_weighted_price'].rolling(window=sma_window).mean()
    
    # Intraday Volatility Adjustment
    df['intraday_volatility'] = df['high'] - df['low']
    df['adjusted_sma'] = df['sma_volume_weighted_price'] - df['intraday_volatility']
    
    # Volume Surge Indicator
    df['volume_change'] = df['volume'] - df['volume'].shift(1)
    volume_surge_threshold = 50000  # Example threshold, adjust as needed
    df['volume_surge'] = (df['volume_change'] > volume_surge_threshold).astype(int)
    
    # Short-Term Momentum
    short_term_window = 4
    df['avg_close_4_days'] = df['close'].rolling(window=short_term_window).mean()
    df['current_momentum'] = df['close'] - df['avg_close_4_days']
    
    # True Average Price
    df['true_avg_price'] = (df['high'] + df['low'] + df['close']) / 3
    
    # Long-Term Price Momentum
    long_term_window = 20
    df['avg_close_20_days'] = df['close'].rolling(window=long_term_window).mean()
    df['long_term_momentum'] = df['close'] - df['avg_close_20_days']
    
    # Difference between Short and Long-Term Momentum
    df['momentum_diff'] = df['current_momentum'] - df['long_term_momentum']
    
    # Calculate Daily Return
    df['daily_return'] = df['close'] / df['close'].shift(1) - 1
