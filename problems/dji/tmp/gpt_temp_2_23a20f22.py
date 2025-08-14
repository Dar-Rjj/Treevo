def heuristics_v2(df):
    # Calculate Volume-Weighted Daily Returns
    df['Daily_Return'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    df['Volume_Weighted_Return'] = df['Daily_Return'] * df['volume']
    
    # Exponential Smoothing
    df['Smoothed_Return'] = df['Volume_Weighted_Return'].ewm(span=5, adjust=False).mean()
    
    # Short-Term and Long-Term Exponential Moving Average of Volume-Weighted Daily Returns
    short_window = 5
    long_window = 20
    df['Short_Term_EMA'] = df['Volume_Weighted_Return'].ewm(span=short_window, adjust=False).mean()
    df['Long_Term_EMA'] = df['Volume_Weighted_Return'].ewm(span=long_window, adjust=False).mean()
    
    # Compute the Extended Dynamic Difference
    df['Extended_Dynamic_Difference'] = df['Short_Term_EMA'] - df['Long_Term_EMA']
    
    # Calculate Weighted High-Low Spread
    df['High_Low_Spread'] = (df['high'] - df['low']) * df['volume']
    
    # Apply Conditional Weight to High-Low Spread
    positive_return_weight = 1.5
