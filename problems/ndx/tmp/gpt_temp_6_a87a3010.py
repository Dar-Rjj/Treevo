def heuristics_v2(df):
    # Calculate the 50-day and 200-day moving averages of the volume
    volume_50 = df['volume'].rolling(window=50).mean()
    volume_200 = df['volume'].rolling(window=200).mean()
    
    # Calculate the volume ratio
    volume_ratio = volume_50 / volume_200
    
    # Calculate the logarithmic return over the last 50 days
    log_return = np.log(df['close'] / df['close'].shift(50))
    
    # Generate the heuristic matrix by multiplying the volume ratio with the logarithmic return
    heuristics_matrix = volume_ratio * log_return
    
    return heuristics_matrix
