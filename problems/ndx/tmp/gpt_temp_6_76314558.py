def heuristics_v2(df):
    log_return = np.log(df['close']).diff()
    avg_price = (df['high'] + df['low']) / 2
    heuristic_values = log_return - avg_price
    smoothed_values = heuristic_values.ewm(span=21, adjust=False).mean()
    skewness_insight = heuristic_values.rolling(window=10).skew()
    heuristics_matrix = smoothed_values + skewness_insight
    return heuristics_matrix
