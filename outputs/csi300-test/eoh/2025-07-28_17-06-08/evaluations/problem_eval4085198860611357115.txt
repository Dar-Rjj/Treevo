def heuristics_v2(df):
    df['ema_20_close'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_50_close'] = df['close'].ewm(span=50, adjust=False).mean()
    df['log_volume_10_sum'] = df['volume'].apply(lambda x: np.log(x+1)).rolling(window=10).sum()
    heuristics_matrix = (df['ema_20_close'] / df['ema_50_close'] * df['log_volume_10_sum']).dropna()
    return heuristics_matrix
