import pandas as pd
    
    # Momentum factor: 5-day return
    momentum = df['close'].pct_change(periods=5).shift(1)
    
    # Volatility factor: 20-day standard deviation of daily returns
    volatility = df['close'].pct_change().rolling(window=20).std().shift(1)
    
    # Volume growth factor: 5-day percentage change in volume
    volume_growth = df['volume'].pct_change(periods=5).shift(1)
    
    # Combining the factors into a single DataFrame
    heuristics_matrix = pd.DataFrame({'Momentum': momentum, 'Volatility': volatility, 'Volume_Growth': volume_growth}).dropna()
    
    return heuristics_matrix
