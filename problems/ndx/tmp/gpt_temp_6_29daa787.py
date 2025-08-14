def heuristics_v2(df):
    # Calculate the 10-day moving average of closing prices
    ma_10 = df['close'].rolling(window=10).mean()
    
    # Calculate the ratio of the current closing price to its 10-day moving average
    close_to_ma_ratio = df['close'] / ma_10
    
    # Calculate the daily return
    daily_return = df['close'].pct_change()
    
    # Calculate the natural logarithm of the daily return
    log_daily_return = np.log(1 + daily_return)
    
    # Combine the ratio and the logarithm of the daily return into a single factor
    combined_factor = close_to_ma_ratio * log_daily_return
    
    # Apply a 20-day exponential moving average to smooth the factor
    smoothed_factor = combined_factor.ewm(span=20, adjust=False).mean()
    
    heuristics_matrix = smoothed_factor.dropna()
    return heuristics_matrix
