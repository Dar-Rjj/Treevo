import numpy as np
    # Calculate the log returns
    log_returns = np.log(df['close'] / df['close'].shift(1))
    
    # Calculate the 10-day rolling standard deviation of the log returns
    std_dev = log_returns.rolling(window=10).std()
    
    # Calculate the 5-day exponentially weighted moving average of the volume
    ewm_volume = df['volume'].ewm(span=5, adjust=False).mean()
    
    # Construct the heuristics matrix
    heuristics_matrix = 0.6 * std_dev + 0.4 * ewm_volume
    
    return heuristics_matrix
