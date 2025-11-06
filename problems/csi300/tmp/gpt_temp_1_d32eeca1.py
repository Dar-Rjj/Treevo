def heuristics_v2(df):
    heuristics_matrix = (df['close'] / df['close'].shift(20) - 1) / (df['high'].rolling(20).std() + 1e-6) * (df['volume'].shift(1) / df['volume'].rolling(20).mean())
    return heuristics_matrix
