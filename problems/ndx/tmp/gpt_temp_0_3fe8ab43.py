import pandas as pd

def heuristics_v2(df):
    # Calculate daily returns
    daily_returns = df['close'].pct_change()
    # Calculate 20-day EMA of daily returns
    ema_daily_returns = daily_returns.ewm(span=20, adjust=False).mean()
    # Calculate the ratio of max to min volume over the last 30 days
    volume_ratio = df['volume'].rolling(window=30).max() / df['volume'].rolling(window=30).min()
    # Compute the heuristic factor
    heuristic_factor = ema_daily_returns * volume_ratio
    return heuristics_matrix
