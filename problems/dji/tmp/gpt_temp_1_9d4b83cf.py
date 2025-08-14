import pandas as pd
    # Calculate the True Range
    df['true_range'] = df[['high', 'low']].apply(lambda x: x['high'] - x['low'], axis=1)
    df['true_range'] = df['true_range'].combine(df['close'].shift(1) - df['low'], max)
    df['true_range'] = df['true_range'].combine(df['close'].shift(1) - df['high'], max)
    
    # Calculate the 10-day average true range
    avg_true_range = df['true_range'].rolling(window=10).mean()
    
    # Calculate the 30-day rolling mean of the closing price
    rolling_mean_close = df['close'].rolling(window=30).mean()
    
    # Calculate the ratio
    ratio = avg_true_range / rolling_mean_close
    
    # Apply a 15-day rolling standard deviation
    heuristics_matrix = ratio.rolling(window=15).std()
    
    return heuristics_matrix
