def heuristics_v2(df):
    df['daily_return'] = df['close'].pct_change()
    df['std_daily_return_15'] = df['daily_return'].rolling(window=15).std()
    df['log_volume'] = np.log(df['volume'])
    df['sum_log_volume_30'] = df['log_volume'].rolling(window=30).sum()
    heuristics_matrix = df['std_daily_return_15'] * df['sum_log_volume_30']
    return heuristics_matrix
