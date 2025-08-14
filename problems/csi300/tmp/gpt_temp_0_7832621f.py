def heuristics_v2(df):
    # Define the lookback window for calculations
    window_size = 30
    
    # Calculate the weighted moving average (WMA) of the closing price
    weights = np.arange(1, window_size + 1)
    df['wma_close'] = df['close'].rolling(window=window_size).apply(lambda prices: np.dot(prices, weights)/weights.sum(), raw=True)
    
    # Calculate the WMA of the trading amount
    df['wma_amount'] = df['amount'].rolling(window=window_size).apply(lambda amounts: np.dot(amounts, weights)/weights.sum(), raw=True)
    
    # Construct the heuristics matrix by combining the WMA of the trading amount and the ratio of the closing price to its WMA
    heuristics_matrix = df['wma_amount'] * (df['close'] / df['wma_close'])
    
    return heuristics_matrix
