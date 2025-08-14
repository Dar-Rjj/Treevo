def heuristics_v2(df):
    min_close_50 = df['close'].rolling(window=50).min()
    avg_volume_10 = df['volume'].rolling(window=10).mean()
    heuristics_matrix = (df['close'] / min_close_50) - np.log(avg_volume_10)
    return heuristics_matrix
