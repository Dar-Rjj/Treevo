def heuristics_v2(df):
    df['wma_close'] = df['close'].rolling(window=14).apply(lambda x: (x * np.arange(1, 15)).sum() / np.arange(1, 15).sum(), raw=True)
    df['volume_ratio'] = df['volume'].rolling(window=7).mean() / df['volume'].rolling(window=28).mean()
    df['atr'] = (df['high'].rolling(window=14).max() - df['low'].rolling(window=14).min()) / 2
    heuristics_matrix = (df['wma_close'].pct_change().shift(-1) * df['volume_ratio']) / df['atr']
    return heuristics_matrix
