def heuristics_v2(df):
    # Calculate the ratio of the high price to the low price
    price_ratio = df['high'] / df['low']
    # Multiply by the logarithm of the volume
    factor_values = price_ratio * np.log(df['volume'])
    # Apply a simple moving average for smoothing the factor values
    smoothed_factor = factor_values.rolling(window=10).mean()
    heuristics_matrix = smoothed_factor.dropna()
    return heuristics_matrix
