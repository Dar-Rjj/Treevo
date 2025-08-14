def heuristics_v2(df):
    price_ratio = df['high'] / df['low']
    log_volume = np.log(df['volume'])
    combined_values = price_ratio * log_volume
    heuristics_matrix = combined_values.ewm(span=14, adjust=False).mean()
    return heuristics_matrix
