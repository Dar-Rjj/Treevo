def heuristics_v2(df):
    # Calculate the 10-day high and low
    high_10 = df['high'].rolling(window=10).max()
    low_10 = df['low'].rolling(window=10).min()

    # Calculate the difference between 10-day high and low
    diff_high_low = high_10 - low_10

    # Calculate the logarithmic return of the trading volume
    log_volume_ret = np.log(df['volume']).diff(periods=1)

    # Combine the two components with weights
    heuristics_matrix = (diff_high_low * 0.7) + (log_volume_ret * 0.3)

    return heuristics_matrix
