def heuristics_v4(df):
    log_returns = np.log(df['close'] / df['close'].shift(1))
    ma_log_return_20d = log_returns.rolling(window=20).mean()
    ema_volume_50d = df['volume'].ewm(span=50, adjust=False).mean()
    heuristics_matrix = ma_log_return_20d / ema_volume_50d
    return heuristics_matrix
