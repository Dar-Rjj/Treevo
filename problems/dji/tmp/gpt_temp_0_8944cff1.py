def heuristics_v2(df, n=20, m=5, k=3, volume_window=5, liquidity_window=5, range_window=5, ema_span=10):
    # Calculate daily return based on close price
