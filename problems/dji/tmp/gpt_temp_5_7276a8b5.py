import pandas as pd
    
    # Define rolling window sizes
    short_window = 10
    long_window = 50
    
    # Calculate simple moving averages
    sma_short = df['close'].rolling(window=short_window).mean()
    sma_long = df['close'].rolling(window=long_window).mean()
    
    # Calculate price change and volume change
    price_change = df['close'].pct_change()
    volume_change = df['volume'].pct_change()
    
    # Calculate relative strength index (RSI) for a 14-day period
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    # Combine all factors into a single score
    heuristics_matrix = (sma_short - sma_long) + price_change + volume_change + rsi
    
    return heuristics_matrix
