def heuristics_v2(df):
    # Calculate the 14-day and 28-day Exponential Moving Averages of close price
    ema_close_14 = df['close'].ewm(span=14, adjust=False).mean()
    ema_close_28 = df['close'].ewm(span=28, adjust=False).mean()
    
    # Calculate the ratio of the two EMAs
    ratio_ema = ema_close_14 / ema_close_28
    
    # Calculate the 14-day standard deviation of daily log returns
    log_returns = np.log(df['close'] / df['close'].shift(1))
    std_dev_log = log_returns.rolling(window=14).std()
    
    # Adjust the EMA ratio by the standard deviation of log returns
    adjusted_ratio = ratio_ema * std_dev_log
    
    # Create the heuristics matrix
    heuristics_matrix = adjusted_ratio.dropna()
    
    return heuristics_matrix
