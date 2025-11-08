def heuristics_v2(df):
    price_accel_short = df['close'].pct_change(periods=2) - df['close'].pct_change(periods=1)
    price_accel_long = df['close'].pct_change(periods=5) - df['close'].pct_change(periods=3)
    volume_trend = df['volume'].rolling(window=5).apply(lambda x: (x[-1] - x[0]) / (x.mean() + 1e-8))
    
    momentum_divergence = price_accel_short - price_accel_long
    volume_adjusted_divergence = momentum_divergence * volume_trend
    
    heuristics_matrix = volume_adjusted_divergence
    
    return heuristics_matrix
