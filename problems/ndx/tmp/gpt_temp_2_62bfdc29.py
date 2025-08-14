def heuristics_v2(df):
    def calculate_volatility(df, window=20):
        return df['close'].rolling(window=window).std()

    def calculate_weighted_volume(df, window=10):
        weights = np.arange(1, window + 1)
        return df['volume'].rolling(window=window).apply(lambda x: (x * weights).sum() / weights.sum(), raw=True)

    # Calculate factors
    volatility = calculate_volatility(df)
    weighted_volume = calculate_weighted_volume(df)

    # Combine factors
    heuristics_matrix = 0.7 * volatility + 0.3 * weighted_volume

    return heuristics_matrix
