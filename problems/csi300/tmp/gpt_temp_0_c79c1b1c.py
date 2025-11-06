def heuristics_v2(df):
    close = df['close']
    volume = df['volume']
    
    momentum_acceleration = (close.pct_change(5) - close.pct_change(10)) * close.pct_change(5)
    volume_persistence = volume.rolling(window=10).apply(lambda x: (x[-1] > x[:-1].mean()).astype(int))
    volatility_scaling = close.rolling(window=20).std()
    
    heuristics_matrix = momentum_acceleration * volume_persistence / volatility_scaling
    return heuristics_matrix
