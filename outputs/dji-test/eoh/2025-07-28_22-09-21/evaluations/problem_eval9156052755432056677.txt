def heuristics_v2(df):
    df['price_change'] = df['close'].pct_change()
    df['wma_close'] = df['close'].rolling(window=10).apply(lambda x: (x * [0.1, 0.15, 0.2, 0.25, 0.3]).sum(), raw=True)
    q1 = df['volume'].rolling(window=20).quantile(0.25)
    q3 = df['volume'].rolling(window=20).quantile(0.75)
    iqr = q3 - q1
    df['volume_rsi'] = 100 - (100 / (1 + (df['volume'] / iqr)))
    df['atr'] = df['high'].rolling(window=10).max() - df['low'].rolling(window=10).min()
    heuristics_matrix = (df['price_change'].shift(-1) * df['volume_rsi'] * df['wma_close']) / df['atr']
    return heuristics_matrix
