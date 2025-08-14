import pandas as pd

def heuristics_v2(df):
    # Calculate daily returns
    daily_returns = df['close'].pct_change()
    # Count the number of positive daily returns
    positive_days = (daily_returns > 0).rolling(window=10).sum()
    # Calculate the rolling total number of trading days
    total_days = daily_returns.rolling(window=10).count()
    # Compute the rolling heuristic factor
    heuristic_factor = (positive_days / total_days).ewm(span=20, adjust=False).mean().fillna(0)
    return heuristics_matrix
