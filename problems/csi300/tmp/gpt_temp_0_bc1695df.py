def heuristics_v2(df):
    heuristics_matrix = (
        (df['close'] / df['close'].shift(20) - 1) * 
        (df['volume'] / df['volume'].rolling(20).mean()) * 
        (1 - (df['high'] - df['low']).rolling(10).std() / df['close'].rolling(10).mean())
    )
    return heuristics_matrix
