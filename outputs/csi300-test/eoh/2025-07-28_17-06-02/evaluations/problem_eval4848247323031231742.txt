import pandas as pd
    
    # Calculate daily return
    daily_return = df['close'].pct_change()
    
    # Calculate momentum over 10 days
    momentum_10d = df['close'] - df['close'].shift(10)
    
    # Calculate average volume over 10 days
    avg_volume_10d = df['volume'].rolling(window=10).mean()
    
    # Calculate volatility (standard deviation) over 10 days
    volatility_10d = df['close'].rolling(window=10).std()
    
    # Heuristic factor: combining momentum, average volume, and inverse of volatility
    heuristics_matrix = (momentum_10d / avg_volume_10d) * (1 / volatility_10d)
    
    return heuristics_matrix
