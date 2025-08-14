def heuristics_v2(df):
    # Calculate the sum of the last 5 days' high prices
    sum_last_5_high = df['high'].rolling(window=5).sum()
    # Calculate the sum of the next 5 days' low prices
    sum_next_5_low = df['low'].shift(-5).rolling(window=5).sum()
    # Calculate the factor as the ratio of the two sums
    factor_values = sum_last_5_high / sum_next_5_low
    # Apply a weighted moving average for smoothing the factor values
    weights = np.array([1, 2, 3, 4, 5])
    smoothed_factor = factor_values.rolling(window=5).apply(lambda x: np.sum(weights * x) / np.sum(weights), raw=True)
    heuristics_matrix = smoothed_factor.dropna()
    return heuristics_matrix
