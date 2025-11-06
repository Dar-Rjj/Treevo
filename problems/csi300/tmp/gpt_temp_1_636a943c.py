def heuristics_v2(df):
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    
    price_position = (close - low.rolling(10).min()) / (high.rolling(10).max() - low.rolling(10).min() + 1e-5)
    volume_acceleration = volume / volume.rolling(20).mean() - 1
    volume_percentile = volume.rolling(20).apply(lambda x: (x[-1] > x.quantile(0.8)).astype(float))
    
    heuristics_matrix = (0.5 - price_position) * volume_acceleration * volume_percentile
    return heuristics_matrix
