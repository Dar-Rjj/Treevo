import pandas as pd
    import talib

    # Calculate MACD
    macd, signal, hist = talib.MACD(df['close'])
    
    # Calculate RSI
    rsi = talib.RSI(df['close'], timeperiod=14)
    
    # Create a basic scoring mechanism for past 5 days performance
    macd_score = macd.diff().rolling(window=5).mean()
    rsi_score = rsi.diff().rolling(window=5).mean()
    
    # Combine scores into a single heuristic
    heuristics_matrix = (macd + macd_score + rsi - rsi_score) / 2
    
    return heuristics_matrix
