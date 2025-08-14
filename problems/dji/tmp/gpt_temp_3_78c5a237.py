import pandas as pd

def heuristics_v2(df):
    # Calculate daily returns
    daily_returns = df['close'].pct_change()
    # 7-day moving standard deviation of daily returns
    std_dev = daily_returns.rolling(window=7).std()
    # 10-day moving average of volume
    avg_volume = df['volume'].rolling(window=10).mean()
    # Log difference between volume and its 10-day moving average
    log_diff = (df['volume'] / avg_volume).apply(lambda x: x if x > 0 else 1e-6).apply(np.log)
    # 20-day exponentially weighted moving average of the log difference
    ewma_log_diff = log_diff.ewm(span=20, adjust=False).mean()
    # Combine the two components
    heuristics_matrix = (std_dev + ewma_log_diff) / 2
    
    return heuristics_matrix
