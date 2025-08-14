def heuristics_v2(df):
    # Calculate the 14-day Exponential Moving Average (EMA) of closing prices
    ema = df['close'].ewm(span=14, adjust=False).mean()
    
    # Calculate the 14-day True Range (TR)
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate the 14-day Average True Range (ATR)
    atr = tr.rolling(window=14).mean()
    
    # Create a heuristic score by combining EMA and ATR
    heuristics_matrix = (ema / atr).dropna()
    
    return heuristics_matrix
