import pandas as pd
    
    # Calculate short-term and long-term moving averages for closing prices
    df['SMA_10'] = df['close'].rolling(window=10).mean()
    df['LMA_50'] = df['close'].rolling(window=50).mean()
    
    # Calculate the difference between short-term and long-term moving averages
    df['MA_Diff'] = df['SMA_10'] - df['LMA_50']
    
    # Calculate the 5-day percentage change in closing price for momentum
    df['Momentum'] = df['close'].pct_change(periods=5)
    
    # Subtract a weighted trading volume from the combined factors
    df['Volume_Weighted'] = df['volume'].rolling(window=10).mean() / df['volume'].rolling(window=50).mean()
    df['Heuristic_Factor'] = (df['MA_Diff'] + df['Momentum']) - df['Volume_Weighted']
    
    # Drop rows with NaN values resulting from rolling calculations
    heuristics_matrix = df['Heuristic_Factor'].dropna()
    
    return heuristics_matrix
