import pandas as pd

def heuristics_v2(df):
    # Calculate daily returns
    daily_returns = df['close'].pct_change()
    # Calculate the cumulative sum of daily returns
    cumsum_daily_returns = daily_returns.cumsum()
    # Calculate the rolling standard deviation of volume with a window of 10 days
    volume_std = df['volume'].rolling(window=10).std()
    # Adjust the cumulative sum of daily returns by the rolling standard deviation of volume
    adjusted_cumsum_returns = cumsum_daily_returns / volume_std
    # Calculate the True Range
    tr = pd.Series(index=df.index)
    tr['TR1'] = df['high'] - df['low']
    tr['TR2'] = (df['high'] - df['close'].shift()).abs()
    tr['TR3'] = (df['low'] - df['close'].shift()).abs()
    true_range = tr.max(axis=1)
    # Calculate the Average True Range over the last 10 days
    atr_10 = true_range.rolling(window=10).mean()
    # Compute the heuristic factor
    heuristic_factor = adjusted_cumsum_returns.mean() / atr_10.mean() if atr_10.mean() > 0 else 0
    # Create a Series for the heuristic factor, repeating it for each date in the DataFrame
    heuristics_matrix = pd.Series([heuristic_factor]*len(df), index=df.index)
    return heuristics_matrix
