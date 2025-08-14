def heuristics_v2(df):
    # Calculate the 30-day and 100-day simple moving averages of the close price
    sma_30 = df['close'].rolling(window=30).mean()
    sma_100 = df['close'].rolling(window=100).mean()
    
    # Calculate the ratio of the 30-day SMA to the 100-day SMA
    sma_ratio = sma_30 / sma_100
    
    # Calculate the daily trading range and its logarithm
    trading_range_log = (df['high'] - df['low']).apply(lambda x: 0 if x == 0 else math.log(x))
    
    # Generate the heuristic matrix by multiplying the SMA ratio with the logarithm of the trading range
    heuristics_matrix = sma_ratio * trading_range_log
    
    return heuristics_matrix
