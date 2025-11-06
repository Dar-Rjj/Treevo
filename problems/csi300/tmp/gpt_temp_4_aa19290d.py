def heuristics_v2(df):
    heuristics_matrix = (df['open'] - df['close'].shift(1)) / df['close'].shift(1) - (df['high'].rolling(6).max() - df['open']) / df['open']
    return heuristics_matrix
