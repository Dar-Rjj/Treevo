import pandas as pd

def heuristics_v2(df):
    # Calculate daily returns
    daily_returns = df['close'].pct_change()
    # Calculate the 14-day EMA of daily volumes
    volume_ema = df['volume'].ewm(span=14).mean()
    # Calculate the 14-day EMA of daily returns
    returns_ema = daily_returns.ewm(span=14).mean()
    # Compute the heuristic factor using EMA
    heuristic_factor = (returns_ema / volume_ema).replace([np.inf, -np.inf], 0).fillna(0)
    return heuristics_matrix
