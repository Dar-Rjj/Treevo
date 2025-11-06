def heuristics_v2(df):
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    
    daily_range = (high - low) / close
    range_quantile = daily_range.rolling(20).apply(lambda x: (x[-1] > x.quantile(0.7)) * 1.0)
    volume_acceleration = volume / volume.shift(1) - volume.shift(1) / volume.shift(2)
    price_momentum = (close - close.shift(5)) / close.shift(5)
    
    heuristics_matrix = range_quantile * price_momentum * volume_acceleration
    return heuristics_matrix
