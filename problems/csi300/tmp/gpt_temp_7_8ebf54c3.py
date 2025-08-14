def heuristics_v2(df):
    window_size = 15
    df['ema_close'] = df['close'].ewm(span=window_size, adjust=False).mean()
    df['wma_close'] = df['close'].rolling(window=window_size).apply(lambda x: (x * pd.Series(range(1, len(x)+1))).sum() / pd.Series(range(1, len(x)+1)).sum(), raw=True)
    df['roc_ema_close'] = df['ema_close'].pct_change()
    df['ema_amount'] = df['amount'].ewm(span=window_size, adjust=False).mean()
    df['ema_volume'] = df['volume'].ewm(span=window_size, adjust=False).mean()
    heuristics_matrix = df['roc_ema_close'] + (df['ema_amount'] - df['ema_volume']) * (df['close'] / df['wma_close'])
    return heuristics_matrix
