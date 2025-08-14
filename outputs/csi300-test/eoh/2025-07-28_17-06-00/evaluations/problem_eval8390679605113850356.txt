def heuristics_v2(df):
    # Calculate 20-day high and low
    df['High_20'] = df['high'].rolling(window=20).max()
    df['Low_20'] = df['low'].rolling(window=20).min()
    # Compute the percentage distance from 20-day high and low
    df['Dist_from_High'] = (df['close'] - df['High_20']) / df['High_20'] * 100
    df['Dist_from_Low'] = (df['Low_20'] - df['close']) / df['Low_20'] * 100
    # Calculate the weighted moving average of volume
    weights = np.arange(1, 6)[::-1]  # Weights for the last 5 days
    df['WMA_volume'] = df['volume'].rolling(window=5).apply(lambda x: np.dot(x, weights), raw=True) / 15  # Normalize weights
    # Combine the distances and WMA of volume for the heuristics matrix
    heuristics_matrix = (df['Dist_from_High'] + df['Dist_from_Low']) * df['WMA_volume']
    return heuristics_matrix
