def heuristics_v2(df):
    high = df['high']
    low = df['low']
    close = df['close']
    open_price = df['open']
    volume = df['volume']
    
    prev_close = close.shift(1)
    overnight_return = (open_price - prev_close) / prev_close
    intraday_range = (high - low) / open_price
    
    volume_trend = volume.rolling(5).apply(lambda x: (x[-1] - x[0]) / x[0] if x[0] != 0 else 0)
    volatility_breakout = intraday_range / intraday_range.rolling(5).mean()
    
    factor = overnight_return * volatility_breakout * volume_trend
    
    heuristics_matrix = factor
    return heuristics_matrix
