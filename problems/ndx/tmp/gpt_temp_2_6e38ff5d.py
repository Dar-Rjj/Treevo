def heuristics_v2(df):
    # Calculate the average of the last 10 days' high prices
    avg_last_10_high = df['high'].rolling(window=10).mean()
    # Calculate the average of the next 10 days' low prices
    avg_next_10_low = df['low'].shift(-10).rolling(window=10).mean()
    # Calculate the factor as the ratio of the two averages
    factor_values = avg_last_10_high / avg_next_10_low
    # Apply a weighted moving average for smoothing the factor values
    weights = np.arange(1, 11)
    smoothed_factor = factor_values.rolling(window=10).apply(lambda x: np.average(x, weights=weights), raw=True)
    heuristics_matrix = smoothed_factor.dropna()
    return heuristics_matrix
