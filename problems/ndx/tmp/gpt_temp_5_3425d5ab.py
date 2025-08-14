def heuristics_v2(df):
    # Calculate the 120-day moving average of closing prices
    ma_120 = df['close'].rolling(window=120).mean()
    
    # Compute the deviation from the 120-day moving average
    deviation = df['close'] - ma_120
    
    # Calculate the 30-day moving standard deviation of log returns
    log_returns = np.log(df['close']).diff()
    volatility = log_returns.rolling(window=30).std()
    
    # Combine the deviation with the volatility
    combined_factor = deviation * volatility
    
    # Apply a 5-day exponential moving average to smooth the factor
    smoothed_factor = combined_factor.ewm(span=5, adjust=False).mean()
    
    heuristics_matrix = smoothed_factor.dropna()
    return heuristics_matrix
