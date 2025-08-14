def heuristics_v2(df):
    df['momentum'] = df['close'].pct_change(periods=5)
    df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    heuristics_matrix = (df['momentum'].shift(-1) * df['vwap']) / df['atr']
    return heuristics_matrix
