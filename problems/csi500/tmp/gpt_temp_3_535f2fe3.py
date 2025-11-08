def heuristics_v2(df):
    price_efficiency = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    vwap = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    efficiency_momentum = price_efficiency - price_efficiency.rolling(window=3).mean()
    vwap_deviation = df['close'] / vwap - 1
    acceleration_signal = efficiency_momentum.rolling(window=5).apply(lambda x: (x[-1] - x[0]) / (abs(x).mean() + 1e-8))
    heuristics_matrix = acceleration_signal * vwap_deviation * df['volume'].rolling(window=10).mean()
    return heuristics_matrix
