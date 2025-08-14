def heuristics_v2(df):
    log_returns = np.log(df['close']).diff()
    vol_moving_avg = df['volume'].rolling(window=30).mean()
    vol_ratio = df['volume'] / vol_moving_avg
    heuristic_values = log_returns * vol_ratio
    heuristics_matrix = heuristic_values.ewm(span=10, adjust=False).mean().dropna()
    return heuristics_matrix
