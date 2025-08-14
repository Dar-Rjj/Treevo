def heuristics_v2(df):
    close_prices = df['close']
    
    # Calculate the Average True Range (ATR)
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - close_prices.shift(1))
    low_close = abs(df['low'] - close_prices.shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr_14 = tr.rolling(window=14).mean()
    
    # Calculate the Rate of Change (ROC) of the closing prices over a 10-day period
    roc_10 = close_prices.pct_change(periods=10)
    
    heuristics_matrix = roc_10 / atr_14
    
    return heuristics_matrix
