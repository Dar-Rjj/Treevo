def heuristics_v2(df):
    # Calculate the sum of the last 5 days' closing prices
    sum_last_5_close = df['close'].rolling(window=5).sum()
    # Calculate the sum of the next 5 days' low prices
    sum_next_5_low = df['low'].shift(-5).rolling(window=5).sum()
    # Calculate the factor as the ratio of the two sums
    factor_values = sum_last_5_close / sum_next_5_low
    # Apply an exponential moving average for smoothing the factor values
    ema_factor = factor_values.ewm(span=10, adjust=False).mean()
    # Apply a weighted moving average for further smoothing
    weights = np.array([0.1, 0.2, 0.3, 0.2, 0.1, 0.1])
    wma_factor = ema_factor.rolling(window=6).apply(lambda x: np.sum(weights * x), raw=True)
    heuristics_matrix = wma_factor.dropna()
    return heuristics_matrix
