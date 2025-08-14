def heuristics_v2(df):
    # Calculate the daily logarithmic return
    log_return = np.log(df['close'] / df['close'].shift(1))
    
    # Calculate the 20-day moving average of volume
    vol_ma_20 = df['volume'].rolling(window=20).mean()
    
    # Compute the ratio of volume to its 20-day moving average
    vol_ratio = df['volume'] / vol_ma_20
    
    # Apply a custom heuristic to combine the logarithmic return and the volume ratio
    heuristics_matrix = (log_return + vol_ratio).rank(pct=True)
    
    return heuristics_matrix
