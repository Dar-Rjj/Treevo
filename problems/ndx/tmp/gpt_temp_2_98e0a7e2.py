def heuristics_v2(df):
    # Calculate Long-Term Momentum (20-day MA)
    long_term_momentum = df['close'].rolling(window=20).mean()
    
    # Calculate Short-Term Momentum (5-day MA)
    short_term_momentum = df['close'].rolling(window=5).mean()
    
    # Calculate Price Momentum
    price_momentum = short_term_momentum - long_term_momentum
    
    # Detect Volume Spike
    volume_10_day_ma = df['volume'].rolling(window=10).mean()
    volume_spike_indicator = (df['volume'] > 2.0 * volume_10_day_ma).astype(int)
    
    # Combine Price Momentum and Volume Spike
    combined_price_momentum = price_momentum * volume_spike_indicator
    
    # Calculate Intraday Volatility
    intraday_volatility = df['high'] - df['low']
    
    # Calculate Intraday Return
    intraday_return = (df['high'] - df['low']) / df['low']
