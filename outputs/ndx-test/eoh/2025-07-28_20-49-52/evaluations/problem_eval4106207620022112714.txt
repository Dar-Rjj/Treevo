def heuristics_v2(df):
    # Calculate the average of the last 5 days' high and low prices
    avg_high_low = (df['high'].rolling(window=5).mean() + df['low'].rolling(window=5).mean()) / 2
    # Calculate the factor as the ratio of the average high and low prices to the current day's close price
    factor_values = avg_high_low / df['close']
    # Apply a weighted moving average for smoothing the factor values
    weights = np.arange(1, 11)
    smoothed_factor = factor_values.rolling(window=10).apply(lambda x: np.average(x, weights=weights), raw=True)
    heuristics_matrix = smoothed_factor.dropna()
    return heuristics_matrix
