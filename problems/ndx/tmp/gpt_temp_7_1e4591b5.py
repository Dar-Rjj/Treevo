def heuristics_v2(df):
    log_median_price = (df['high'].apply(np.log) + df['low'].apply(np.log)) / 2
    weight = df['volume'] / df['volume'].sum()
    weighted_avg = (df['open'] * weight + df['close'] * weight).cumsum()
    heuristic_values = log_median_price - weighted_avg
    heuristics_matrix = heuristic_values.ewm(span=10, adjust=False).mean().dropna()
    return heuristics_matrix
