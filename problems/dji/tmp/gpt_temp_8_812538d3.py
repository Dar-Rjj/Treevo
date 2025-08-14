import numpy as np
    # Calculate the log-returns
    log_returns = np.log(df['close'] / df['close'].shift(1))
    
    # Compute the 60-day rolling standard deviation of log-returns for volatility
    std_log_returns = log_returns.rolling(window=60).std()
    
    # Calculate the 120-day minimum close price
    min_close_120 = df['close'].rolling(window=120).min()
    
    # Compute the distance (difference) from the current close to the 120-day min
    dist_to_min = (df['close'] - min_close_120) / min_close_120
    
    # Calculate the percentage change in volume and its 30-day rolling std for volatility
    vol_change = df['volume'].pct_change()
    std_vol_change = vol_change.rolling(window=30).std()
    
    # Heuristics matrix combining all factors
    heuristics_matrix = std_log_returns + dist_to_min - std_vol_change
    
    return heuristics_matrix
