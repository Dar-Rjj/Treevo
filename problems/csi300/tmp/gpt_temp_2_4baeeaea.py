import pandas as pd
    
    # Calculate the difference between the rolling max of 'high' and the rolling min of 'low'
    diff_high_low = df['high'].rolling(window=5).max() - df['low'].rolling(window=5).min()
    
    # Calculate the average volume over a rolling window
    avg_volume = df['volume'].rolling(window=5).mean()
    
    # Generate the heuristics matrix
    heuristics_matrix = diff_high_low / avg_volume
    
    return heuristics_matrix
