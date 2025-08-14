import numpy as np

    # Calculate 10-day standard deviation of the closing price
    std_10 = df['close'].rolling(window=10).std()
    
    # Calculate log returns
    log_returns = np.log(df['close'] / df['close'].shift(1))
    
    # Calculate 50-day EWMA of the log return
    ewma_50_log_returns = log_returns.ewm(span=50, adjust=False).mean()
    
    # Calculate 20-day SMA of the high and low prices
    sma_20_high = df['high'].rolling(window=20).mean()
    sma_20_low = df['low'].rolling(window=20).mean()
    
    # Combine factors
    heuristics_matrix = std_10 + ewma_50_log_returns + (sma_20_high - sma_20_low)
    
    return heuristics_matrix
