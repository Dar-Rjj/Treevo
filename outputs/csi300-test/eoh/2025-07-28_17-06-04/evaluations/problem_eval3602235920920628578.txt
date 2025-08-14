def heuristics_v2(df):
    # Calculate the rate of change of volume
    roc_volume = df['volume'].pct_change()
    # Apply a weighted moving average to the rate of change of volume
    wma_roc_volume = roc_volume.rolling(window=21).apply(lambda x: (x * (21 - np.arange(21))).sum() / (21 * 22 / 2), raw=True)
    # Calculate the true range
    tr = df['high'] - df['low']
    # Calculate the exponential moving average of the true range over a 7-day window
    ema_tr = tr.ewm(span=7, adjust=False).mean()
    # Combine the two factors to create a new heuristics matrix
    heuristics_matrix = (wma_roc_volume + ema_tr).dropna()
    return heuristics_matrix
