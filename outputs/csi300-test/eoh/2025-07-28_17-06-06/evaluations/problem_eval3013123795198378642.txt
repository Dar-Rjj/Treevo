import pandas as pd
    import numpy as np
    
    # Calculate the daily returns for each column
    returns = df.pct_change().fillna(0)
    
    # Calculate the 10-day cumulative return for dynamic weighting
    cum_returns = (1 + returns).rolling(window=10).apply(np.prod, raw=True) - 1
    
    # Compute the weights based on the absolute value of 10-day cumulative returns
    weights = cum_returns.abs() / cum_returns.abs().sum(axis=1).values.reshape(-1, 1)
    
    # Compute the heuristic score by applying the weights to the original dataframe
    heuristics_matrix = (df * weights).sum(axis=1)
    
    return heuristics_matrix
