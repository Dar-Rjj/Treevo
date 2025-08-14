def heuristics_v2(df):
    weight = df['close'].shift(1) / df['open']
    pct_change_close = df['close'].pct_change()
    log_volume = np.log(df['volume'])
    heuristics_matrix = (weight * pct_change_close + (1 - weight) * log_volume).dropna()
    return heuristics_matrix
