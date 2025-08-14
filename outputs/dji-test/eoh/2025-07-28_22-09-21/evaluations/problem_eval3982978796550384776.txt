def heuristics_v2(df):
    df['weighted_close'] = (df['close'].rolling(window=14).sum() / 14) * df['close']
    df['ad_line'] = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low']) * df['volume']
    df['mad'] = df['close'].rolling(window=20).apply(lambda x: np.median(np.abs(x - np.median(x))), raw=True)
    heuristics_matrix = (df['weighted_close'] * df['ad_line'].shift(-1)) / df['mad']
    return heuristics_matrix
