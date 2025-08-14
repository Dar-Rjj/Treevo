def heuristics_v2(df):
    df['roc_10'] = df['close'].pct_change(periods=10)
    df['atr_14'] = df[['high', 'low', 'close']].rolling(window=14).apply(lambda x: np.max(x) - np.min(x), raw=True)
    df['volume_to_atr_ratio'] = df['volume'] / df['atr_14']
    heuristics_matrix = df['roc_10'] * df['volume_to_atr_ratio']
    return heuristics_matrix
