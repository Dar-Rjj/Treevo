def heuristics_v2(df):
    close = df['close']
    volume = df['volume']
    high = df['high']
    low = df['low']
    
    # Calculate the 20-day momentum of the closing price
    momentum_close = close - close.shift(20)
    
    # Calculate the average true range (ATR) as a volatility measure
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=20).mean()
    
    # Calculate the 10-day relative strength of the volume
    rsv = (volume - volume.rolling(window=10).min()) / (volume.rolling(window=10).max() - volume.rolling(window=10).min())
    
    # Alpha factor as a combination of momentum, ATR, and RSV
    heuristics_matrix = (momentum_close / atr) * rsv
    
    return heuristics_matrix
