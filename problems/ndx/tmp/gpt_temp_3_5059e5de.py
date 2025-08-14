def heuristics_v2(df):
    # Calculate the 14-day weighted moving average (WMA) of closing prices
    weights = pd.Series(range(1, 15))
    wma_close = df['close'].rolling(window=14).apply(lambda x: (x * weights).sum() / weights.sum(), raw=True)
    
    # Calculate the 14-day weighted moving average (WMA) of low prices
    wma_low = df['low'].rolling(window=14).apply(lambda x: (x * weights).sum() / weights.sum(), raw=True)
    
    # Calculate the difference between WMA of close and WMA of low
    diff_wma = wma_close - wma_low
    
    # Calculate the 14-day momentum of the closing prices
    momentum = df['close'] - df['close'].shift(14)
    
    # Adjust the difference by the momentum
    adjusted_diff = diff_wma * momentum
    
    # Create the heuristics matrix
    heuristics_matrix = adjusted_diff.dropna()
    
    return heuristics_matrix
