def heuristics_v2(df):
    # Calculate the average of the last 10 days' volume
    avg_last_10_volume = df['volume'].rolling(window=10).mean()
    # Calculate the average of the next 10 days' closing prices
    avg_next_10_close = df['close'].shift(-10).rolling(window=10).mean()
    # Calculate the factor as the ratio of the two averages
    factor_values = avg_last_10_volume / avg_next_10_close
    # Apply a weighted moving average for smoothing the factor values
    weights = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    smoothed_factor = factor_values.rolling(window=10).apply(lambda x: np.dot(x, weights)/weights.sum(), raw=True)
    heuristics_matrix = smoothed_factor.dropna()
    return heuristics_matrix
