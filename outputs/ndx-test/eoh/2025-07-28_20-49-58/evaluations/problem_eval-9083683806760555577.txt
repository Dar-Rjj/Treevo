import pandas as pd

def heuristics_v2(df):
    # Calculate daily return
    df['daily_return'] = df['close'].pct_change()
    
    # Calculate price volatility (standard deviation of daily returns)
    df['volatility_5'] = df['daily_return'].rolling(window=5).std()
    df['volatility_10'] = df['daily_return'].rolling(window=10).std()
    
    # Calculate moving averages
    df['ma_5'] = df['close'].rolling(window=5).mean()
    df['ma_10'] = df['close'].rolling(window=10).mean()
    
    # Calculate volume change
    df['volume_change'] = df['volume'].pct_change()
    
    # Heuristic factor: ratio of 5-day to 10-day volatility
    df['heuristic_1'] = df['volatility_5'] / df['volatility_10']
    
    # Heuristic factor: difference between 5-day and 10-day moving averages
    df['heuristic_2'] = df['ma_5'] - df['ma_10']
    
    # Heuristic factor: product of volume change and 5-day volatility
    df['heuristic_3'] = df['volume_change'] * df['volatility_5']
    
    # Combine all heuristic factors into a single matrix
    heuristics_matrix = df[['heuristic_1', 'heuristic_2', 'heuristic_3']]
    
    return heuristics_matrix
