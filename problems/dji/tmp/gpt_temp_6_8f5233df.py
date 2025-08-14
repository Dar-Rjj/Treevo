import pandas as pd
    
    # Calculate the 20-day average of closing prices
    df['20_day_avg'] = df['close'].rolling(window=20).mean()
    
    # Calculate the raw momentum factor
    df['momentum_factor'] = df['close'] - df['20_day_avg']
    
    # Apply a 5-day Simple Moving Average (SMA) to smooth the momentum factor
    heuristics_matrix = df['momentum_factor'].rolling(window=5).mean()
    
    return heuristics_matrix
