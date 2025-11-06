def heuristics_v2(df):
    high, low, close, volume = df['high'], df['low'], df['close'], df['volume']
    typical_price = (high + low + close) / 3
    volume_weighted_price = (typical_price * volume).rolling(window=10).sum() / volume.rolling(window=10).sum()
    volatility_breakout = (high - low).rolling(window=5).std()
    intraday_reversal = (close - typical_price.shift(1)) / typical_price.shift(1)
    heuristics_matrix = (volume_weighted_price - close) / volatility_breakout * intraday_reversal
    return heuristics_matrix
