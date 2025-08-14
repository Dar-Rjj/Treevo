def heuristics_v2(df):
    # Relative Strength Index (RSI)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    sma_20 = df['close'].rolling(window=20).mean()
    std_20 = df['close'].rolling(window=20).std()
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    bb_width = (upper_band - lower_band) / sma_20
    
    # Aroon Oscillator
    aroon_up = 100 * (20 - df['high'].rolling(window=20).apply(lambda x: x.argmax())) / 19
    aroon_down = 100 * (df['low'].rolling(window=20).apply(lambda x: x.argmin()) / 19)
    aroon_oscillator = aroon_up - aroon_down
    
    # Volume Momentum Indicator
    volume_mom = df['volume'].pct_change(12)
    
    # Composite heuristic
    heuristics_matrix = (rsi + bb_width + aroon_oscillator + volume_mom) / 4
    return heuristics_matrix
