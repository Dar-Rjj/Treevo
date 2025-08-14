def heuristics_v2(df):
    # Calculate the 10-day Weighted Moving Average (WMA) of close prices
    weights = np.arange(1, 11)
    wma_close = df['close'].rolling(window=10).apply(lambda x: (x * weights).sum() / weights.sum(), raw=True)
    
    # Calculate the 20-day Exponential Moving Average (EMA) of volume
    ema_volume = df['volume'].ewm(span=20, adjust=False).mean()
    
    # Create a heuristic score by combining WMA of close prices and EMA of volume
    heuristics_matrix = (wma_close * ema_volume).dropna()
    
    return heuristics_matrix
