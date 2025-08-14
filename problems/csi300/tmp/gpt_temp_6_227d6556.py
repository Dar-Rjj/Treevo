def heuristics_v2(df):
    window_size = 10
    df['ema_close'] = df['close'].ewm(span=window_size, adjust=False).mean()
    df['ema_volume'] = df['volume'].ewm(span=window_size, adjust=False).mean()
    weights = np.arange(1, window_size+1)
    wma_close = df['close'].rolling(window=window_size).apply(lambda x: (x*weights).sum() / weights.sum(), raw=True)
    adjustment_factor = (df['close'] - wma_close) / df['close']
    heuristics_matrix = df['ema_volume'] * (df['close'] / df['ema_close']) * (1 + adjustment_factor)
    return heuristics_matrix
