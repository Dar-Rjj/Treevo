def heuristics_v2(df):
    # Calculate Volume-Weighted Price
    df['volume_weighted_price'] = df['close'] * df['volume']
    
    # EMA of Volume-Weighted Prices
    df['ema_volume_weighted_price'] = df['volume_weighted_price'].ewm(span=20, adjust=False).mean()
    
    # Intraday Volatility Adjustment
    df['intraday_volatility'] = df['high'] - df['low']
    df['adjusted_ema'] = df['ema_volume_weighted_price'] + df['intraday_volatility']
    
    # Volume Spike Indicator
    df['volume_change'] = df['volume'].diff()
    df['volume_spike'] = (df['volume_change'] > 3 * df['volume'].std()).astype(int)
    df['final_alpha_factor'] = df['adjusted_ema']
    df.loc[df['volume_spike'] == 1, 'final_alpha_factor'] *= 1.5
    
    # Short-Term Momentum
    df['short_term_momentum'] = df['close'].rolling(window=5).sum()
    
    # Long-Term Momentum
    df['long_term_momentum'] = df['close'].rolling(window=20).sum()
    
    # Difference between Short and Long-Term Momentum
    df['momentum_difference'] = df['short_term_momentum'] - df['long_term_momentum']
    
    # Incorporate Volume Trend
    df['short_term_volume'] = df['volume'].rolling(window=5).sum()
    df['long_term_volume'] = df['volume'].rolling(window=20).sum()
    df['volume_trend_difference'] = df['short_term_volume'] - df['long_term_volume']
    
    # Calculate Adjusted High-Low Range
    df['adjusted_high_low_range'] = (df['high'] - df['low']) * (df['volume'] / df['volume'].rolling(window=20).sum())
    
    # Previous Day's Return
    df['previous_day_return'] = (df['close'] - df['open'].shift(1)) / df['open'].shift(1)
