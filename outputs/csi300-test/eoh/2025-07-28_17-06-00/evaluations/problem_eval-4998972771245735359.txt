def heuristics_v2(df):
    window_size = 20
    df['ema_close'] = df['close'].ewm(span=window_size, adjust=False).mean()
    df['ema_volume'] = df['volume'].ewm(span=window_size, adjust=False).mean()
    mad_close = df['close'].rolling(window=window_size).apply(lambda x: np.fabs(x - x.mean()).mean(), raw=True)
    weight_close_ratio = mad_close / (mad_close + df['volume'].rolling(window=window_size).apply(lambda x: np.fabs(x - x.mean()).mean(), raw=True))
    heuristics_matrix = (df['close'] / df['ema_close']) * weight_close_ratio + df['ema_volume'] * (1 - weight_close_ratio)
    return heuristics_matrix
