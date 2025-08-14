import pandas as pd
    
    # Define a rolling window size
    window_size = 20
    
    # Calculate rolling mean and standard deviation for each feature
    rolling_mean = df.rolling(window=window_size).mean()
    rolling_std = df.rolling(window=window_size).std()
    
    # Compute RSI for the 'close' price
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window_size).mean()
    avg_loss = loss.rolling(window=window_size).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Weighted combination to form the heuristics matrix
    weight_rolling_mean = 0.4
    weight_rolling_std = 0.3
    weight_rsi = 0.3
    heuristics_matrix = (weight_rolling_mean * rolling_mean + 
                         weight_rolling_std * rolling_std + 
                         weight_rsi * rsi.reindex(rolling_mean.index, method='bfill'))
    
    # Return the heuristics matrix
    return heuristics_matrix
