def heuristics_v2(df):
    df['price_change'] = df['close'].pct_change()
    df['dm_plus'] = np.where(df['high'] > df['high'].shift(1), df['high'] - df['high'].shift(1), 0)
    df['dm_minus'] = np.where(df['low'].shift(1) > df['low'], df['low'].shift(1) - df['low'], 0)
    df['tr'] = df[['high' - 'low', 'high' - 'close'].shift(1), 'close'].shift(1) - df['low']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=14).mean()
    df['di_plus'] = (df['dm_plus'].rolling(window=14).sum() / df['atr']) * 100
    df['di_minus'] = (df['dm_minus'].rolling(window=14).sum() / df['atr']) * 100
    df['dx'] = (np.abs(df['di_plus'] - df['di_minus']) / (df['di_plus'] + df['di_minus'])) * 100
    df['adx'] = df['dx'].rolling(window=14).mean()
    heuristics_matrix = df['adx']
    return heuristics_matrix
