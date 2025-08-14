import pandas as pd
    import numpy as np
    
    n = 5  # Number of days for return calculation
    m = 20  # Number of days for volatility (standard deviation) calculation
    
    # Calculate the n-day return
    df['n_day_return'] = df['close'].pct_change(n)
    
    # Calculate the m-day volatility (standard deviation of returns)
    df['m_day_volatility'] = df['close'].pct_change().rolling(window=m).std()
    
    # Composite Heuristics Factor: High return + Low volatility
    df['heuristics_factor'] = df['n_day_return'] / (df['m_day_volatility'] + 1e-8)  # Adding a small number to avoid division by zero
    
    heuristics_matrix = df['heuristics_factor'].dropna()
    
    return heuristics_matrix
