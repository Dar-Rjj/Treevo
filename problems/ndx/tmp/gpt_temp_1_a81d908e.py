def heuristics_v2(df):
    # Calculate the rate of change of prices over a 10-day period
    roc = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    
    # Compute the Average True Range (ATR) as a measure of volatility
    tr = df[['high'-'low', 'high'-'close'.shift(), 'low'-'close'.shift()]].max(axis=1)
    atr = tr.rolling(window=14).mean()
    
    # Adjust the volume using the ATR
    smoothed_volume = df['volume'].ewm(span=10, adjust=False).mean() / atr
    
    # Integrate the rate of change and the ATR-adjusted, smoothed volume
    heuristics_matrix = (roc + smoothed_volume).rank(pct=True)
    
    return heuristics_matrix
