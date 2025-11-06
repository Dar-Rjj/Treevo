def heuristics_v2(df):
    high, low, volume = df['high'], df['low'], df['volume']
    
    range_ratio = (high - low) / (high.rolling(window=10).mean() + low.rolling(window=10).mean())
    volume_ma = volume.rolling(window=15).mean()
    asymmetric_vol = range_ratio.rolling(window=5, min_periods=1).apply(lambda x: x[x > x.mean()].mean() if len(x[x > x.mean()]) > 0 else 0)
    
    heuristics_matrix = asymmetric_vol * (volume / volume_ma)
    return heuristics_matrix
