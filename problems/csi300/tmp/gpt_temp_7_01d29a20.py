def heuristics_v2(df):
    df['ema_20d_close'] = df['close'].ewm(span=20, adjust=False).mean()
    df['min_vol_30d'] = df['volume'].rolling(window=30).min()
    df['vol_ratio'] = df['volume'] / df['min_vol_30d']
    df['log_vol_ratio'] = np.log(df['vol_ratio'])
    df['composite_score'] = df['ema_20d_close'] * df['log_vol_ratio']
    heuristics_matrix = df['composite_score'].dropna()
    return heuristics_matrix
