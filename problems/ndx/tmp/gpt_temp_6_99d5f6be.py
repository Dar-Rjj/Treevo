def heuristics_v2(df):
    # Calculate the maximum of the last 20 days' high prices
    max_last_20_high = df['high'].rolling(window=20).max()
    # Calculate the minimum of the next 20 days' low prices
    min_next_20_low = df['low'].shift(-20).rolling(window=20).min()
    # Calculate the factor as the ratio of the two calculated values
    factor_values = max_last_20_high / min_next_20_low
    # Apply a weighted moving average for smoothing the factor values
    weights = np.arange(1, 11)
    smoothed_factor = factor_values.rolling(window=10).apply(lambda x: np.sum(weights * x) / np.sum(weights), raw=False)
    heuristics_matrix = smoothed_factor.dropna()
    return heuristics_matrix
