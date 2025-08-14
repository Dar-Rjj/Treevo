def heuristics_v2(df):
    # Calculate the 5-day logarithmic return of closing prices
    log_return = np.log(df['close']).diff(5)
    # Calculate the 5-day average daily trading volume
    avg_volume = df['volume'].rolling(window=5).mean()
    # Calculate the factor as the ratio of the 5-day log return to the 5-day average volume
    factor_values = log_return / avg_volume
    # Apply a weighted moving average for smoothing the factor values
    weights = np.arange(1, 6)
    smoothed_factor = factor_values.rolling(window=5).apply(lambda x: np.average(x, weights=weights), raw=True)
    heuristics_matrix = smoothed_factor.dropna()
    return heuristics_matrix
