def heuristics_v2(df):
    df['log_diff'] = (df['close'] - df['open']).apply(lambda x: np.log(x + 1))
    df['weighted_log_diff'] = df['log_diff'] * df['volume']
    heuristics_matrix = df['weighted_log_diff'].rolling(window=20).sum()
    return heuristics_matrix
