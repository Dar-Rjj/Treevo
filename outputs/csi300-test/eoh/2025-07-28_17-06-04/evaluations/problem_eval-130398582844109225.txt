import pandas as pd
    
    # Calculate short-term and long-term simple moving averages
    short_window = 10
    long_window = 50
    df['SMA_short'] = df['close'].rolling(window=short_window, min_periods=1).mean()
    df['SMA_long'] = df['close'].rolling(window=long_window, min_periods=1).mean()
    
    # Relative Strength (RS)
    df['RS'] = df['close'] / df['close'].shift(1)
    
    # Weighted combination based on historical predictive power
    # These weights are illustrative and should be determined through backtesting
    w_sma_short = 0.4
    w_sma_long = -0.3
    w_rs = 0.6
    
    # Generate the heuristics matrix for each date
    heuristics_matrix = w_sma_short * df['SMA_short'] + w_sma_long * df['SMA_long'] + w_rs * df['RS']
    
    return heuristics_matrix
