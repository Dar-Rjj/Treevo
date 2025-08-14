def heuristics_v2(df):
    df['price_momentum'] = df['close'].pct_change(periods=14)
    df['volume_mean'] = df['volume'].rolling(window=14).mean()
    df['volume_diff_score'] = (df['volume'] - df['volume_mean']) / df['volume_mean']
    df['atr'] = df[['high', 'low', 'close']].shift(1).join(df['close']).apply(lambda x: max(x[0], x[1]) - min(x[0], x[1]), axis=1).rolling(window=14).mean()
    heuristics_matrix = df['price_momentum'] * df['volume_diff_score'] / df['atr']
    return heuristics_matrix
