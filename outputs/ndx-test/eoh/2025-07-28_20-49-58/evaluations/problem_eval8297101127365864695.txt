import pandas as pd
    
    # Calculate MACD
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    macd_line = ema_12 - ema_26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd = macd_line - signal_line

    # Calculate RSI
    delta = df['close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # Calculate 5-day EMA of volume
    ema_volume = df['volume'].ewm(span=5, adjust=False).mean()

    # Combine factors
    heuristics_matrix = macd * rsi * ema_volume

    return heuristics_matrix
