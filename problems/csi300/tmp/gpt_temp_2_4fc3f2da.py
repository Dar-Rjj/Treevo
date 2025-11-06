def heuristics_v2(df):
    heuristics_matrix = (df['open'] - df['close'].shift(1)) / (df['high'].shift(1) - df['low'].shift(1)) * (df['volume'] / df['volume'].rolling(window=5).mean())
    return heuristics_matrix
