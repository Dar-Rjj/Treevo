def heuristics_v2(df):
    # Calculate Intraday Volatility
    df['intraday_volatility'] = df['high'] - df['low']
    
    # Calculate Long-Term Momentum (25-day Simple Moving Average)
    df['long_term_momentum'] = df['close'].rolling(window=25).mean()
    
    # Calculate Short-Term Momentum (7-day Simple Moving Average)
    df['short_term_momentum'] = df['close'].rolling(window=7).mean()
    
    # Calculate Price Momentum
    df['price_momentum'] = df['short_term_momentum'] - df['long_term_momentum']
    
    # Detect Volume Surge
    df['volume_change'] = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1) * 100
    df['volume_surge_indicator'] = (df['volume_change'] > 60).astype(int)
    
    # Combine Price Momentum and Volume Surge
    df['combined_momentum'] = df['price_momentum'] * df['volume_surge_indicator']
    df['sum_combined_momentum'] = df['combined_momentum'].rolling(window=10).sum()
    
    # Adjust Combined Momentum by Intraday Volatility
    df['adjusted_momentum'] = df['sum_combined_momentum'].abs() / df['intraday_volatility']
    
    # Calculate Intraday Return
    df['intraday_return'] = (df['high'] - df['low']) / df['low']
