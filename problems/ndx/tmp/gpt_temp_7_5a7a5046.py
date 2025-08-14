def heuristics_v2(df):
    # Calculate the 20-day high and low
    high_20 = df['high'].rolling(window=20).max()
    low_20 = df['low'].rolling(window=20).min()
    
    # Compute the ratio of the current close price to the 20-day high and low
    high_ratio = df['close'] / high_20
    low_ratio = df['close'] / low_20
    
    # Calculate the daily change in trading volume
    vol_change = df['volume'].pct_change().fillna(0)
    
    # Logarithm of the absolute value of daily trading volume change, adding 1 to avoid log(0)
    log_vol_change = (vol_change + 1).apply(lambda x: abs(x)).apply(np.log)
    
    # Combine the ratios and log volume change into a single heuristic
    heuristics_matrix = 0.5 * (high_ratio - low_ratio) + 0.5 * log_vol_change
    
    return heuristics_matrix
