def heuristics_v2(df):
    # Calculate the rate of change of volume
    roc_volume = df['volume'].pct_change()
    # Apply an exponential moving average to the rate of change of volume
    ema_roc_volume = roc_volume.ewm(span=14, adjust=False).mean()
    # Calculate the true range
    tr = df['high'] - df['low']
    # Define weights for the weighted moving average
    weights = np.arange(1, 11)
    # Calculate the weighted moving average of the true range
    wma_tr = tr.rolling(window=10).apply(lambda x: np.sum(x * weights) / np.sum(weights), raw=True)
    # Combine the two factors to create a new heuristics matrix
    heuristics_matrix = (ema_roc_volume + wma_tr).dropna()
    return heuristics_matrix
