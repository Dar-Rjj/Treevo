import pandas as pd

def heuristics_v2(df):
    # Calculate short-term and long-term moving averages for the close price
    df['SMA_10'] = df['close'].rolling(window=10).mean()
    df['LMA_50'] = df['close'].rolling(window=50).mean()
    # Momentum calculation based on the difference between today's close and the close 10 days ago
    df['Momentum_10'] = df['close'] - df['close'].shift(10)
    # Volume change relative to the average volume over the last 10 days
    df['Volume_Change'] = (df['volume'] - df['volume'].rolling(window=10).mean()) / df['volume'].rolling(window=10).mean()
    # Construct the heuristic matrix by combining the above metrics
    heuristics_matrix = df[['SMA_10', 'LMA_50', 'Momentum_10', 'Volume_Change']].dropna()
    # Returning the heuristic values as a Series with multi-index (date, metric)
    return heuristics_matrix
