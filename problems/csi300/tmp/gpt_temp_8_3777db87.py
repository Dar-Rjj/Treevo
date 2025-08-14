def heuristics_v2(df):
    window_size = 20
    df['ema_close'] = df['close'].ewm(span=window_size, adjust=False).mean()
    df['sma_volume'] = df['volume'].rolling(window=window_size).mean()
    daily_return = df['close'].pct_change().fillna(0)
    std_daily_return = daily_return.rolling(window=window_size).std()
    mean_daily_return = daily_return.rolling(window=window_size).mean()
    cv_daily_return = std_daily_return / mean_daily_return
    weight_vol_ratio = 1 / (1 + cv_daily_return)
    heuristics_matrix = (np.log(df['close'] / df['ema_close'])) * weight_vol_ratio + df['sma_volume'] * (1 - weight_vol_ratio)
    return heuristics_matrix
