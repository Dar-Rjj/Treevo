def heuristics_v2(df):
    # Calculate the maximum of the last 7 days' high prices
    max_last_7_high = df['high'].rolling(window=7).max()
    # Calculate the minimum of the next 7 days' low prices
    min_next_7_low = df['low'].shift(-7).rolling(window=7).min()
    # Calculate the factor as the difference between the two
    factor_values = max_last_7_high - min_next_7_low
    # Apply a weighted moving average for smoothing the factor values
    weights = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3])
    smoothed_factor = factor_values.rolling(window=7).apply(lambda x: (x * weights).sum(), raw=True)
    heuristics_matrix = smoothed_factor.dropna()
    return heuristics_matrix
