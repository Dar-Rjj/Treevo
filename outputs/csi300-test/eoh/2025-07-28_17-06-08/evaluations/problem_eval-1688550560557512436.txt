import pandas as pd
    
    # Calculate daily return
    df['daily_return'] = df['close'].pct_change()
    
    # Determine if the volume is increasing compared to the previous day
    df['volume_increase'] = (df['volume'] > df['volume'].shift(1)).astype(int)
    
    # Find days where the return is positive
    df['positive_return'] = (df['daily_return'] > 0).astype(int)
    
    # Calculate the rolling sum of days with both positive return and increasing volume over a window
    window_size = 20
    df['heuristic_factor'] = (df['positive_return'] * df['volume_increase']).rolling(window=window_size).sum()
    
    # Extract the heuristics series
    heuristics_matrix = df['heuristic_factor'].copy()
    
    return heuristics_matrix
