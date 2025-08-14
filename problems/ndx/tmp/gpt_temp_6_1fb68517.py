def heuristics_v2(df):
    heuristics_matrix = (df['high'].rolling(window=10).mean() - df['low'].rolling(window=10).mean()) / df['adj close']
    return heuristics_matrix
