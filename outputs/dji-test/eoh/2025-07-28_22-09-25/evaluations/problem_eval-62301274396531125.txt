def heuristics_v2(df):
    def momentum(df, window):
        return df['close'].pct_change(window)
    
    def volatility(df, window):
        return df['close'].pct_change().rolling(window=window).std()
    
    def volume_swing(df, window):
        return (df['volume'] - df['volume'].shift(window)) / df['volume'].shift(window)
    
    def price_range(df, window):
        high = df['high'].rolling(window=window).max()
        low = df['low'].rolling(window=window).min()
        return (df['close'] - low) / (high - low)
    
    # Example time windows
    short_window = 10
    medium_window = 30
    long_window = 90
    
    # Compute individual factors
    momentum_short = momentum(df, short_window)
    momentum_medium = momentum(df, medium_window)
    momentum_long = momentum(df, long_window)
    
    volatility_short = volatility(df, short_window)
    volatility_medium = volatility(df, medium_window)
    volatility_long = volatility(df, long_window)
    
    volume_swing_short = volume_swing(df, short_window)
    volume_swing_medium = volume_swing(df, medium_window)
    volume_swing_long = volume_swing(df, long_window)
    
    price_range_short = price_range(df, short_window)
    price_range_medium = price_range(df, medium_window)
    price_range_long = price_range(df, long_window)
    
    # Combine factors into a DataFrame
    heuristics_matrix = pd.DataFrame({
        'momentum_short': momentum_short,
        'momentum_medium': momentum_medium,
        'momentum_long': momentum_long,
        'volatility_short': volatility_short,
        'volatility_medium': volatility_medium,
        'volatility_long': volatility_long,
        'volume_swing_short': volume_swing_short,
        'volume_swing_medium': volume_swing_medium,
        'volume_swing_long': volume_swing_long,
        'price_range_short': price_range_short,
        'price_range_medium': price_range_medium,
        'price_range_long': price_range_long
    })
    
    # Return the final heuristics matrix
    return heuristics_matrix
