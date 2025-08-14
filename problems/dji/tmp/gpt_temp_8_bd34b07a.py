import pandas as pd
    short_window = 10
    long_window = 50
    vol_window = 20
    
    # Calculate simple moving averages
    sma_short = df['close'].rolling(window=short_window).mean()
    sma_long = df['close'].rolling(window=long_window).mean()
    
    # Calculate standard deviation for volatility
    volatility = df['close'].rolling(window=vol_window).std()
    
    # Alpha factor calculation
    heuristics_matrix = (sma_short - sma_long) / volatility
    
    return heuristics_matrix
