def heuristics_v2(df):
    df['log_return'] = np.log(df['close']) - np.log(df['close'].shift(1))
    df['avg_volume_20'] = df['volume'].rolling(window=20).mean()
    heuristics_matrix = df['log_return'] * (df['volume'] / df['avg_volume_20'])
    return heuristics_matrix
