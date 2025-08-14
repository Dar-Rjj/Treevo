def heuristics_v2(df):
    df['momentum'] = df['close'].pct_change(periods=14)
    df['volume_rate'] = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    df['volume_smoothed'] = df['volume_rate'].ewm(span=14).mean()
    df['atr'] = df[['high' - 'low', 'high' - df['close'].shift(1), df['close'].shift(1) - 'low']].abs().max(axis=1).rolling(window=14).mean()
    heuristics_matrix = (df['momentum'].shift(-1) * df['volume_smoothed']) / df['atr']
    return heuristics_matrix
