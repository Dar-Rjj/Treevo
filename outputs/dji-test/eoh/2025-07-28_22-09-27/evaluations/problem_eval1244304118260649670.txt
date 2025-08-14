def heuristics_v2(df):
    log_volume_price_ratio = (df['volume'] / df['close']).apply(np.log)
    heuristics_matrix = log_volume_price_ratio.ewm(span=20, adjust=False).mean()
    return heuristics_matrix
