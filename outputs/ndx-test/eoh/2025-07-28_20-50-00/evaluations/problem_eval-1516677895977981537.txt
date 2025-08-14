import pandas as pd

def heuristics_v2(df):
    # Calculate daily returns
    daily_returns = df['close'].pct_change()
    # Calculate the inverse square root of the trading volume
    inv_sqrt_volume = 1 / (df['volume']**0.5)
    # Calculate the weighted moving average of daily returns
    wma_daily_returns = (daily_returns * inv_sqrt_volume).rolling(window=10).sum() / inv_sqrt_volume.rolling(window=10).sum()
    # Calculate the standard deviation of the last 5 days' daily returns
    std_last_5_days = daily_returns.rolling(window=5).std().fillna(0)
    # Compute the heuristic factor
    heuristic_factor = wma_daily_returns.mean() / std_last_5_days if std_last_5_days.any() > 0 else 0
    # Create a Series for the heuristic factor, repeating it for each date in the DataFrame
    heuristics_matrix = pd.Series([heuristic_factor]*len(df), index=df.index)
    return heuristics_matrix
