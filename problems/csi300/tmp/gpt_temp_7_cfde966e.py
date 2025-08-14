def heuristics_v2(df):
    close = df['close']
    volume = df['volume']
    high = df['high']
    low = df['low']

    # Simple Moving Average for closing price
    sma_close = close.rolling(window=10).mean()
    
    # Percentage change in volume over a 10-day period
    pct_change_volume = volume.pct_change(periods=10)
    
    # Logarithm of the ratio of highest to lowest price within a 20-day lookback window
    log_high_low_ratio = (high.rolling(window=20).max() / low.rolling(window=20).min()).apply(np.log)
    
    # Alpha factor as a combination of SMA close, % change in volume, and log of high-low ratio
    heuristics_matrix = (sma_close + pct_change_volume) * log_high_low_ratio
    
    return heuristics_matrix
