def heuristics_v2(df):
    # Calculate the average of the last 5 days' high prices
    avg_last_5_high = df['high'].rolling(window=5).mean()
    # Calculate the standard deviation of the next 5 days' volume
    std_next_5_vol = df['volume'].shift(-5).rolling(window=5).std()
    # Calculate the factor as the product of the two values
    factor_values = avg_last_5_high * std_next_5_vol
    # Apply a weighted moving average for smoothing the factor values
    weights = np.array([0.1, 0.2, 0.3, 0.2, 0.1])
    smoothed_factor = factor_values.rolling(window=5).apply(lambda x: np.sum(weights * x), raw=True)
    heuristics_matrix = smoothed_factor.dropna()
    return heuristics_matrix
