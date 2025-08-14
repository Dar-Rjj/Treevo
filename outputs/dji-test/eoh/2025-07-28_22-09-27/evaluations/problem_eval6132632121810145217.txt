import pandas as pd
    
    # Calculate simple moving averages
    sma_5 = df['close'].rolling(window=5).mean()
    sma_20 = df['close'].rolling(window=20).mean()
    
    # Momentum calculation
    momentum = df['close'] / df['close'].shift(20) - 1
    
    # Volatility (standard deviation of daily returns over the last 20 days)
    daily_returns = df['close'].pct_change()
    volatility = daily_returns.rolling(window=20).std()
    
    # Composite heuristic matrix
    heuristics_matrix = (sma_5 - sma_20) + momentum - volatility
    
    return heuristics_matrix
