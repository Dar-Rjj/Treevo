def heuristics_v2(df):
    # Calculate the 20-day percentage change in volume
    volume_change = df['volume'].pct_change(20)
    
    # Calculate the 10-day rate of change in closing prices
    close_roc = df['close'].diff(10) / df['close'].shift(10)
    
    # Compute the raw alpha factor as the product of volume change and rate of change in closing prices
    raw_alpha_factor = volume_change * close_roc
    
    # Apply a 5-day moving average to smooth the raw alpha factor
    smoothed_alpha_factor = raw_alpha_factor.rolling(window=5).mean()
    
    # Return the heuristics matrix
    return heuristics_matrix
