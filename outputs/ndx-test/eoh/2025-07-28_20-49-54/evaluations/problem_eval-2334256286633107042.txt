def heuristics_v2(df):
    # Calculate the 30-period exponential moving average of the closing price
    ema_30 = df['close'].ewm(span=30, adjust=False).mean()
    
    # Calculate the 10-period average volume
    avg_volume_10 = df['volume'].rolling(window=10).mean()
    
    # Calculate the volume ratio (current volume / 10-day average volume)
    volume_ratio = df['volume'] / avg_volume_10
    
    # Apply logarithm to the volume ratio
    log_volume_ratio = volume_ratio.apply(np.log)
    
    # Combine the EMA and log of volume ratio
    heuristics_matrix = (ema_30 + log_volume_ratio).rank(pct=True)
    
    return heuristics_matrix
