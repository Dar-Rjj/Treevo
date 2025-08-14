import pandas as pd

def heuristics_v2(df):
    # Calculate daily returns
    daily_returns = df['close'].pct_change()
    # Calculate the mean of the daily volumes
    volume_mean = df['volume'].mean()
    # Calculate the standard deviation of daily returns
    std_daily_returns = daily_returns.std()
    # Compute the heuristic factor
    heuristic_factor = std_daily_returns / (1 + np.log(volume_mean)) if volume_mean > 0 else 0
    # Create a Series for the heuristic factor, repeating it for each date in the DataFrame
    heuristics_matrix = pd.Series([heuristic_factor]*len(df), index=df.index)
    return heuristics_matrix
