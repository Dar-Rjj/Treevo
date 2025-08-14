def heuristics_v2(df):
    log_returns = np.log(df['close'] / df['close'].shift(1)).fillna(0)
    volume_ratio = df['volume'] / df['volume'].rolling(window=50).mean()
    channel_high = df['high'].rolling(window=20).max()
    channel_low = df['low'].rolling(window=20).min()
    price_distance = (df['close'] - channel_low) / (channel_high - channel_low)
    heuristics_matrix = log_returns * volume_ratio * price_distance
    return heuristics_matrix
