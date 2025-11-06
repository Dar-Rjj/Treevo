def heuristics_v2(df):
    high, low, close, volume = df['high'], df['low'], df['close'], df['volume']
    
    # Dynamic momentum with volatility adjustment
    vol_window = 20
    vol = high.rolling(vol_window).std()
    momentum_short = close.pct_change(5)
    momentum_long = close.pct_change(20)
    vol_adjusted_momentum = (momentum_short - momentum_long) / (vol + 1e-8)
    
    # Volume-based reversal signal
    volume_ma = volume.rolling(10).mean()
    volume_spike = (volume - volume_ma) / (volume_ma + 1e-8)
    price_reversal = -close.pct_change(3) * volume_spike
    
    # Range efficiency factor
    true_range = high.combine(close.shift(1), max) - low.combine(close.shift(1), min)
    range_efficiency = (close - close.shift(5)) / (true_range.rolling(5).sum() + 1e-8)
    
    # Combined alpha factor
    heuristics_matrix = vol_adjusted_momentum + price_reversal + range_efficiency
    
    return heuristics_matrix
