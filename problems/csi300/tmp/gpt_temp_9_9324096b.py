import pandas as pd
    
    # Calculate daily returns
    df['daily_return'] = df['close'].pct_change()
    
    # Calculate 10-day moving average of daily returns
    df['avg_return_10d'] = df['daily_return'].rolling(window=10).mean()
    
    # Calculate 10-day average volume
    df['avg_volume_10d'] = df['volume'].rolling(window=10).mean()
    
    # Generate heuristic score
    df['heuristic_score'] = df['avg_return_10d'] - (df['avg_volume_10d'] / df['avg_volume_10d'].max())
    
    heuristics_matrix = df['heuristic_score'].dropna()
    
    return heuristics_matrix
