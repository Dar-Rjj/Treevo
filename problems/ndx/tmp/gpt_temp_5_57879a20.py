import pandas as pd
    
    # Calculate daily returns
    returns = df['close'].pct_change()
    
    # Calculate momentum factors for different periods
    momentum_5d = df['close'].pct_change(5)
    momentum_10d = df['close'].pct_change(10)
    momentum_20d = df['close'].pct_change(20)
    
    # Calculate volatility
    volatility = returns.rolling(window=20).std()
    
    # Combine momentum factors with a volatility adjustment
    heuristics_matrix = (momentum_5d + momentum_10d + momentum_20d) / (volatility + 1e-6)
    
    return heuristics_matrix
