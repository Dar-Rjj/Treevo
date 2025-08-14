import pandas as pd
    
    # Calculate daily returns
    returns = df['close'].pct_change().dropna()
    
    # Calculate rolling standard deviation (volatility) for each feature
    volatilities = df.rolling(window=30).std()
    
    # Adjust each feature by its corresponding volatility
    adjusted_features = df / volatilities
    
    # Define weights for the features
    weights = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    # Calculate the weighted sum of the adjusted features
    heuristics_matrix = (adjusted_features * weights).sum(axis=1)
    
    return heuristics_matrix
