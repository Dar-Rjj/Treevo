def heuristics_v2(df):
    # Short-term and long-term simple moving averages of the close prices
    short_window = 10
    long_window = 30
    df['SMA_short'] = df['close'].rolling(window=short_window).mean()
    df['SMA_long'] = df['close'].rolling(window=long_window).mean()
    
    # Difference between the two moving averages for detecting trends
    df['SMA_diff'] = df['SMA_short'] - df['SMA_long']
    
    # Estimate historical volatility using the standard deviation of daily returns
