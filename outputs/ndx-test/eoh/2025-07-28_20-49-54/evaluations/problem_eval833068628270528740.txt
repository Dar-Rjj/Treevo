import pandas as pd

    # Compute short-term (5 days) and long-term (30 days) moving averages for each feature
    short_mavg = df.rolling(window=5).mean()
    long_mavg = df.rolling(window=30).mean()
    
    # Calculate the difference between short-term and long-term moving averages
    diff_mavg = short_mavg - long_mavg
    
    # Compute the rate of change over 5 periods for the original data
    roc_5 = df.pct_change(periods=5)
    
    # Combine the difference in moving averages with the rate of change to form the heuristics matrix
    heuristics_matrix = (diff_mavg + roc_5).sum(axis=1)  # Summing across columns for simplicity
    
    return heuristics_matrix
