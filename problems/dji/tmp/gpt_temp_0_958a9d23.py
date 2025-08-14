import pandas as pd
    
    # Compute MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()

    # Calculate RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Volume trend
    volume_trend = df['volume'].pct_change()
    
    # Heuristic formula
    heuristics_matrix = (macd - signal) * 0.5 + rsi * 0.3 + (volume_trend * 0.2)
    return heuristics_matrix
