def heuristics_v2(df):
    price_diff = df['high'] - df['low']
    log_close = np.log(df['close'])
    weighted_series = price_diff * log_close
    heuristics_matrix = weighted_series.ewm(span=15, adjust=False).mean()
    return heuristics_matrix
