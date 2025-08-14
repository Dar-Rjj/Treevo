def heuristics_v2(df):
    price_diff = df['close'] - df['open']
    volume_log_change = np.log1p(df['volume'].diff().fillna(0))
    heuristics_matrix = (price_diff * volume_log_change).cumsum()
    return heuristics_matrix
