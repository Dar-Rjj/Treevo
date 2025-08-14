def heuristics_v2(df):
    # Exponential Moving Average (EMA) of Closing Price
    ema_close = df['close'].ewm(span=20, adjust=False).mean()
    
    # True Range
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift())
    tr3 = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Smoothed True Range using EMA
    sma_true_range = true_range.ewm(span=20, adjust=False).mean()
    
    # Heuristics matrix combining EMA and Smoothed True Range
    heuristics_matrix = ema_close + sma_true_range
    return heuristics_matrix
