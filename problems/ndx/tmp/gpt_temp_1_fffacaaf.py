def heuristics_v2(df):
    # Calculate Daily Price Change
    df['price_change'] = df['close'] - df['open']
    
    # Compute 5-Day and 10-Day EMAs of Price Change
    df['ema_5'] = df['price_change'].ewm(span=5, adjust=False).mean()
    df['ema_10'] = df['price_change'].ewm(span=10, adjust=False).mean()
    
    # Determine Reversal Signal
    df['reversal_signal'] = 0
    df.loc[df['ema_5'] > df['ema_10'], 'reversal_signal'] = 1
    df.loc[df['ema_5'] < df['ema_10'], 'reversal_signal'] = -1
    
    # Filter by Volume
    volume_threshold = df['volume'].mean()
    df['filtered_signal'] = df['reversal_signal'] * (df['volume'] > volume_threshold)
    
    # Calculate Volume Adjusted Price (VAP)
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['vap'] = df['typical_price'] * df['volume']
    
    # Calculate Volume Adjusted Momentum
    df['vap_ema_20'] = df['vap'].ewm(span=20, adjust=False).mean()
    df['momentum'] = df['vap'] - df['vap_ema_20']
    
    # Incorporate Recent Volatility
    df['daily_returns'] = df['close'].pct_change()
