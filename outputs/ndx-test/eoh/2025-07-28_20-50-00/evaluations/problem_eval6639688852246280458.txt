import pandas as pd

def heuristics_v2(df):
    # Calculate daily returns
    daily_returns = df['close'].pct_change()
    # Sum of positive daily returns over a 30-day period
    sum_pos_returns = daily_returns[daily_returns > 0].rolling(window=30).sum()
    # Standard deviation of the daily volumes
    volume_std = df['volume'].std()
    # Compute the heuristic factor
    heuristic_factor = sum_pos_returns / volume_std if volume_std > 0 else 0
    # Create a Series for the heuristic factor, repeating it for each date in the DataFrame
    heuristics_matrix = pd.Series([heuristic_factor]*len(df), index=df.index)
    return heuristics_matrix
