def heuristics_v2(df):
    heuristics_matrix = ((np.log(df['close']) - np.log(df['open'])) / df['volume'].rolling(window=10).max()).rolling(window=30).mean()
    return heuristics_matrix
