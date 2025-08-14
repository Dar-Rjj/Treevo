def heuristics_v2(df):
    df['price_change'] = df['close'].pct_change()
    df['volume_wma'] = df['volume'].rolling(window=30).apply(lambda x: (x * pd.Series(range(1, len(x)+1))).sum() / pd.Series(range(1, len(x)+1)).sum(), raw=False)
    df['atr_21'] = ATR(df['high'], df['low'], df['close'], timeperiod=21)
    heuristics_matrix = (df['price_change'].shift(-1) * df['volume_wma']) / df['atr_21']
    return heuristics_matrix
