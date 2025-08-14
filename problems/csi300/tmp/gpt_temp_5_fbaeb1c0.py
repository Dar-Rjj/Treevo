import pandas as pd
    
    # Calculate short-term and long-term simple moving averages
    sma_short = df['close'].rolling(window=10).mean()
    sma_long = df['close'].rolling(window=50).mean()
    
    # Calculate volatility as standard deviation over a period
    vol = df['close'].rolling(window=20).std()
    
    # Compute the heuristics factor
    heuristics_factor = (sma_short - sma_long) / vol
    
    return heuristics_matrix
