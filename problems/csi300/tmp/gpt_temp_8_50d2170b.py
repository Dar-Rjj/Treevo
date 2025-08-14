def heuristics_v2(df):
    df['close_open_ratio'] = df['close'] / df['open']
    df['log_volume'] = df['volume'].apply(lambda x: math.log(x) if x > 0 else 0)
    heuristics_matrix = (df['close_open_ratio'] * df['log_volume']).dropna()
    return heuristics_matrix
