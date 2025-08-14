def heuristics_v2(df):
    df['daily_return'] = df['close'].pct_change()
    df['avg_volume_30'] = df['volume'].rolling(window=30).mean()
    df['vol_ratio'] = df['volume'] / df['avg_volume_30']
    ema_daily_return = df['daily_return'].ewm(span=20, adjust=False).mean()
    log_vol_ratio = np.log(df['vol_ratio'])
    heuristics_matrix = (ema_daily_return - log_vol_ratio).dropna()
    return heuristics_matrix
