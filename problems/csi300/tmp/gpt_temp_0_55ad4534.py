def heuristics_v2(df):
    window_size = 20
    df['ema_close'] = df['close'].ewm(span=window_size, adjust=False).mean()
    log_diff = np.log(df['close']) - np.log(df['ema_close'])
    vol_std = df['volume'].rolling(window=window_size).std()
    avg_vol = df['volume'].rolling(window=window_size).mean()
    heuristics_matrix = log_diff * (vol_std / avg_vol)
    return heuristics_matrix
