def heuristics_v2(df):
    close = df['close']
    volume = df['volume']
    high = df['high']
    low = df['low']
    
    # Calculate the Average True Range (ATR)
    atr = (high - low).rolling(window=10).mean()
    
    # Rate of change in ATR over a 10-day period
    roc_atr = atr.pct_change(periods=10)
    
    # Logarithmic return of the closing price
    log_return = close.pct_change().apply(lambda x: np.log(1 + x))
    
    # Simple Moving Average of the volume
    sma_volume = volume.rolling(window=10).mean()
    
    # Alpha factor as a combination of ROC ATR, logarithmic return, and SMA volume
    heuristics_matrix = (roc_atr + log_return) * sma_volume
    
    return heuristics_matrix
