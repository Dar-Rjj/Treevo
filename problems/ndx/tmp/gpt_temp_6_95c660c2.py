def heuristics(df):
    # Calculate Intraday Volatility
    df['intraday_volatility'] = df['high'] - df['low']
    
    # Calculate Price Momentum
    n_day = 10
    df['price_momentum'] = df['close'] - df['close'].rolling(window=n_day).mean()
    
    # Detect Volume Surge
    df['volume_surge'] = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1) * 100
    volume_surge_threshold = 50
    df['volume_surge_indicator'] = (df['volume_surge'] > volume_surge_threshold).astype(int)
    
    # Adjust Final Factor by Intraday Volatility
    df['adjusted_momentum'] = df['price_momentum'] / df['intraday_volatility']
    
    # Calculate Price Oscillator
    df['price_oscillator'] = (df['high'] - df['low']) / df['close'].shift(1)
    
    # Integrate Adjusted Momentum and Price Oscillator
    df['integrated_momentum_oscillator'] = df['adjusted_momentum'] * df['price_oscillator']
    
    # Incorporate Price Change Rate
    df['price_change_rate'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1) * 100
    df['combined_factor'] = df['integrated_momentum_oscillator'] + df['price_change_rate']
    
    # Calculate Intraday Return
    df['intraday_return'] = (df['high'] - df['low']) / df['low']
