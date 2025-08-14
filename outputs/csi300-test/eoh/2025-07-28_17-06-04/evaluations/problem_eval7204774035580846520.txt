import pandas as pd

def heuristics_v2(df):
    # Calculate the rate of change of closing price
    roc_close = df['close'].pct_change()
    # Apply a weighted moving average to the rate of change of closing price
    wma_roc_close = roc_close.rolling(window=21).apply(lambda x: (x * pd.Series(range(1, 22))).sum() / 210, raw=True)
    # Calculate the daily returns
    daily_returns = df['close'].pct_change()
    # Calculate the standard deviation of the daily returns over a 21-day window
    std_daily_returns = daily_returns.rolling(window=21).std()
    # Combine the two factors to create a new heuristics matrix
    heuristics_matrix = (wma_roc_close + std_daily_returns).dropna()
    return heuristics_matrix
