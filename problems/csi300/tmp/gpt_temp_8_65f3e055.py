import pandas as pd
    import numpy as np
    
    # Calculate daily returns
    df['daily_return'] = df['close'].pct_change()
    
    # Calculate daily volume change
    df['volume_change'] = df['volume'].pct_change()
    
    # Estimate market volatility using standard deviation of daily returns over a 30-day window
    df['volatility'] = df['daily_return'].rolling(window=30).std()
    
    # Adjust weights: higher volatility reduces weight on volume change, assuming more noise in such periods
    df['weight_price'] = 1 - df['volatility'] / df['volatility'].max()
    df['weight_volume'] = 1 - df['weight_price']
    
    # Compute the heuristic factor
    df['heuristic_factor'] = (df['daily_return'] * df['weight_price']) + (df['volume_change'] * df['weight_volume'])
    
    heuristics_matrix = df['heuristic_factor'].copy()
    
    return heuristics_matrix
