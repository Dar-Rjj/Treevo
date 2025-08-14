import pandas as pd
    import numpy as np
    
    # Calculate short-term and long-term moving averages for close prices
    short_window = 10
    long_window = 50
    df['short_mavg'] = df['close'].rolling(window=short_window, min_periods=short_window).mean()
    df['long_mavg'] = df['close'].rolling(window=long_window, min_periods=long_window).mean()
    
    # Calculate rolling standard deviation (volatility) of close prices over a 30-day window
    df['volatility'] = df['close'].rolling(window=30, min_periods=30).std()
    
    # Generate heuristic factor: difference between short and long moving averages normalized by volatility
    df['heuristic_factor'] = (df['short_mavg'] - df['long_mavg']) / df['volatility']
    
    # Prepare the output series
    heuristics_matrix = df['heuristic_factor']
    
    return heuristics_matrix
