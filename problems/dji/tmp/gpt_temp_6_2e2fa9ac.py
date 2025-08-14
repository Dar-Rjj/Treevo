import pandas as pd
    
    # Calculate simple moving averages
    sma_short = df['close'].rolling(window=5).mean()
    sma_long = df['close'].rolling(window=20).mean()
    
    # Calculate momentum
    momentum = df['close'] - df['close'].shift(20)
    
    # Calculate historical volatility
    daily_returns = df['close'].pct_change()
    volatility = daily_returns.rolling(window=20).std()
    
    # Combine factors with weights adjusted by historical volatility
    heuristics_matrix = (sma_short - sma_long) * (momentum / volatility)
    
    return heuristics_matrix
