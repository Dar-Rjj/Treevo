def heuristics_v2(df):
    df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['ema_volume'] = df['volume'].ewm(span=10, adjust=False).mean()
    heuristics_matrix = (df['adx'] * (df['close'] / df['sma_50'])) / df['ema_volume']
    return heuristics_matrix
