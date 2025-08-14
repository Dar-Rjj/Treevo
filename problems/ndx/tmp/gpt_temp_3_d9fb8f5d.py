def heuristics_v2(df):
    # Calculate the 30-day and 90-day weighted moving averages of the close price
    weights_30 = np.arange(1, 31)
    wma_30 = df['close'].rolling(window=30).apply(lambda x: np.sum(weights_30 * x) / np.sum(weights_30), raw=False)
    
    weights_90 = np.arange(1, 91)
    wma_90 = df['close'].rolling(window=90).apply(lambda x: np.sum(weights_90 * x) / np.sum(weights_90), raw=False)
    
    # Calculate the WMA difference
    wma_diff = wma_30 - wma_90
    
    # Calculate the daily return
    df['Return'] = df['close'].pct_change()
    
    # Calculate the 60-day standard deviation of daily returns
    std_60 = df['Return'].rolling(window=60).std()
    
    # Generate the heuristic matrix by dividing the WMA difference with the standard deviation
    heuristics_matrix = wma_diff / std_60
    
    return heuristics_matrix
