def heuristics_v2(df):
    window_size = 20
    df['ema_close'] = df['close'].ewm(span=window_size, adjust=False).mean()
    df['log_return'] = np.log(df['close']) - np.log(df['close'].shift(1))
    df['ema_amount'] = df['amount'].ewm(span=window_size, adjust=False).mean()
    df['ema_volume'] = df['volume'].ewm(span=window_size, adjust=False).mean()
    heuristics_matrix = (df['ema_amount'] / df['ema_volume']) * df['log_return']
    return heuristics_matrix
