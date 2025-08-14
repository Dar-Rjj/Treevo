def heuristics_v2(df):
    log_volume_price_ratio = np.log(df['volume'] / df['close'])
    ema_20 = log_volume_price_ratio.ewm(span=20, adjust=False).mean()
    ema_5 = log_volume_price_ratio.ewm(span=5, adjust=False).mean()
    heuristics_matrix = ema_20 - ema_5
    return heuristics_matrix
