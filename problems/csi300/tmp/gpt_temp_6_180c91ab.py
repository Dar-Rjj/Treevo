def heuristics_v2(df):
    high = df['high']
    low = df['low']
    close = df['close']
    open_ = df['open']
    volume = df['volume']
    
    prev_close = close.shift(1)
    gap = (open_ - prev_close) / prev_close
    gap_rank = gap.rolling(20).apply(lambda x: (x.iloc[-1] > x).mean())
    
    vol_range = (high - low) / close
    volume_momentum = volume / volume.rolling(10).mean()
    
    heuristics_matrix = gap_rank * volume_momentum / (vol_range + 1e-8)
    return heuristics_matrix
