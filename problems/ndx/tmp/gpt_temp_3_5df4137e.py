def heuristics_v2(df):
    # Calculate the 20-day and 60-day exponential moving averages of the close price
    ema_20 = df['close'].ewm(span=20, adjust=False).mean()
    ema_60 = df['close'].ewm(span=60, adjust=False).mean()
    
    # Calculate the ratio between the EMAs
    ema_ratio = ema_20 / ema_60
    
    # Calculate the daily log return
    df['Log_Return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Calculate the 30-day standard deviation of daily log returns
    std_dev_30 = df['Log_Return'].rolling(window=30).std()
    
    # Generate the heuristic matrix by multiplying the EMA ratio with the standard deviation
    heuristics_matrix = ema_ratio * std_dev_30
    
    return heuristics_matrix
