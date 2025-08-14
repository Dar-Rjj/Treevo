def heuristics_v2(df):
    window_size = 14
    close = df['close']
    
    # Exponential Moving Average for closing price
    ema_close = close.ewm(span=window_size, adjust=False).mean()
    
    # Average True Range
    tr = df['high'] - df['low']
    atr = tr.rolling(window=window_size).mean()
    
    # Relative Strength Index (RSI)
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window_size).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window_size).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Hyperbolic tangent transformation and weighted sum
    heuristics_matrix = (close / ema_close * atr) * np.tanh(rsi / 100)
    
    return heuristics_matrix
