import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate daily log returns
    daily_log_returns = np.log(df['close']).diff()
    # Calculate the mean absolute deviation of daily closing prices
    close_mad = df['close'].mad()
    # Calculate the median of daily log returns
    median_daily_log_returns = np.median(daily_log_returns)
    # Compute the heuristic factor
    heuristic_factor = median_daily_log_returns / close_mad if close_mad > 0 else 0
    # Create a Series for the heuristic factor, repeating it for each date in the DataFrame
    heuristics_matrix = pd.Series([heuristic_factor]*len(df), index=df.index)
    return heuristics_matrix
