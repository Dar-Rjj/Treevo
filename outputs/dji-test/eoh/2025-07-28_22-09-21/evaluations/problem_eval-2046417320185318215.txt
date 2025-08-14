def heuristics_v2(df):
    df['price_momentum'] = df['close'].pct_change(periods=21)
    df['volume_roc'] = df['volume'].pct_change(periods=21)
    df['atr'] = df[['high', 'low', 'close']].rolling(window=21).apply(lambda x: np.max(x) - np.min(x), raw=True)
    heuristics_matrix = (df['price_momentum'] * df['volume_roc']) / df['atr']
    return heuristics_matrix
