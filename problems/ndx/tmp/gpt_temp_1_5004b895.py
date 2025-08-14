def heuristics_v2(df):
    # Calculate the logarithmic returns
    log_returns = (df['close'] / df['close'].shift(1)).apply(np.log)
    
    # Sum of the logarithmic returns for the last 10 days
    sum_log_returns_last_10 = log_returns.rolling(window=10).sum()
    
    # Sum of the logarithmic returns for the next 10 days
    sum_log_returns_next_10 = log_returns.shift(-10).rolling(window=10).sum()
    
    # Calculate the factor as the ratio of the two sums
    factor_values = sum_log_returns_last_10 / sum_log_returns_next_10
    
    # Apply an exponential moving average for smoothing the factor values
    smoothed_factor = factor_values.ewm(span=20, adjust=False).mean()
    
    heuristics_matrix = smoothed_factor.dropna()
    return heuristics_matrix
