def heuristics_v2(df):
    # Calculate the factor as the ratio of the sum of high and low prices to the closing price
    factor_values = (df['high'] + df['low']) / df['close']
    # Apply a weighted moving average for smoothing the factor values
    weights = np.arange(1, 11)
    smoothed_factor = factor_values.rolling(window=10).apply(lambda x: np.average(x, weights=weights), raw=True)
    heuristics_matrix = smoothed_factor.dropna()
    return heuristics_matrix
