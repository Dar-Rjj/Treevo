def heuristics_v2(df):
    daily_return = df['close'].pct_change()
    log_volume = df['volume'].apply(lambda x: 0 if x == 0 else np.log(x))
    heuristics_matrix = (daily_return * log_volume).rolling(window=14).sum()
    return heuristics_matrix
