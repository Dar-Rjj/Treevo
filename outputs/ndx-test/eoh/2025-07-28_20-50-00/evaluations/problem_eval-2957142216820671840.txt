def heuristics_v2(df):
    # Calculate 50-day and 200-day simple moving averages of the closing price
    sma_50 = df['close'].rolling(window=50).mean()
    sma_200 = df['close'].rolling(window=200).mean()
    # Calculate the difference between 50-day and 200-day SMAs
    sma_diff = sma_50 - sma_200
    # Calculate the natural logarithm of the ratio of today's volume to yesterday's volume
    volume_ratio_log = (df['volume'] / df['volume'].shift(1)).apply(np.log)
    # Combine SMA difference and log(volume ratio) into a heuristic factor
    heuristics_matrix = (sma_diff + volume_ratio_log).dropna()
    return heuristics_matrix
