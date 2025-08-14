def heuristics_v2(df):
    df['price_momentum'] = df['close'].pct_change(periods=20)
    df['volume_roc'] = df['volume'].pct_change()
    df['true_range'] = df[['high', 'low', 'close']].shift(1).join(df['high']).join(df['low']).apply(lambda x: max(x) - min(x), axis=1)
    df['tr_mean_28'] = df['true_range'].rolling(window=28).mean()
    heuristics_matrix = (df['price_momentum'] * df['volume_roc']) / df['tr_mean_28']
    return heuristics_matrix
