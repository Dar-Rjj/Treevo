def heuristics_v2(df):
    close = df['close']
    volume = df['volume']
    
    short_acceleration = close.pct_change(periods=3) - close.pct_change(periods=6)
    long_trend = close.rolling(window=20).mean() / close.rolling(window=50).mean() - 1
    divergence = short_acceleration - long_trend
    
    volume_persistence = volume.rolling(window=10).apply(lambda x: (x.diff().fillna(0) > 0).sum())
    volume_weight = volume_persistence / 10
    
    heuristics_matrix = divergence * volume_weight
    return heuristics_matrix
