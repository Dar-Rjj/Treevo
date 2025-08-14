def heuristics_v2(df):
    # Exponential Moving Average (EMA) of Closing Price
    ema_close = df['close'].ewm(span=20, adjust=False).mean()
    
    # True Range
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift())
    tr3 = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Heuristics matrix combining EMA and True Range
    heuristics_matrix = ema_close + 0.5 * true_range
    return heuristics_matrix
