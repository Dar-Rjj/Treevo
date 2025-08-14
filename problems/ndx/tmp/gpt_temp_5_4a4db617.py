def heuristics_v2(df):
    log_diff = np.log(df['high']) - np.log(df['low'])
    roc_volume = df['volume'].pct_change(5)
    heuristic_values = log_diff * roc_volume
    heuristics_matrix = heuristic_values.ewm(span=10, adjust=False).mean().dropna()
    return heuristics_matrix
