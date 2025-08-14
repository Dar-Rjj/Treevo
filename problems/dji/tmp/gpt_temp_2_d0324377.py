import pandas as pd

def heuristics_v2(df):
    # Calculate daily log returns
    log_returns = (df['close'] / df['close'].shift(1)).apply(lambda x: math.log(x))
    
    # Calculate 30-day cumulative sum of daily log returns
    cum_sum_log_returns = log_returns.rolling(window=30).sum()
    
    # Calculate 10-day standard deviation of daily log returns
    std_log_returns = log_returns.rolling(window=10).std()
    
    # Create the heuristic matrix
    heuristics_matrix = (cum_sum_log_returns / std_log_returns).fillna(0)
    return heuristics_matrix
