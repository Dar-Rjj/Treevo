def heuristics_v2(df):
    high = df['high']
    low = df['low']
    close = df['close']
    open_ = df['open']
    volume = df['volume']
    
    gap = (open_ - close.shift(1)) / close.shift(1)
    range_prev = high.shift(1) - low.shift(1)
    volume_ratio = volume / volume.rolling(20).mean()
    
    gap_vol_scaled = gap / (range_prev.rolling(10).std() + 1e-5)
    volume_signal = volume_ratio.rolling(5).apply(lambda x: x.rank(pct=True).iloc[-1])
    
    heuristics_matrix = -gap_vol_scaled * volume_signal
    return heuristics_matrix
