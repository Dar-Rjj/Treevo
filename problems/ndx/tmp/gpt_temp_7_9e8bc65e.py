def heuristics_v2(df):
    # Calculate the 60-day rate of change of the close price
    roc_60 = df['close'].pct_change(periods=60)
    
    # Calculate the daily log return
    log_return = np.log(df['close'] / df['close'].shift(1))
    
    # Calculate the 120-day standard deviation of the log returns
    std_120 = log_return.rolling(window=120).std()
    
    # Generate the heuristic matrix by dividing the ROC by the STD
    heuristics_matrix = roc_60 / std_120
    
    return heuristics_matrix
