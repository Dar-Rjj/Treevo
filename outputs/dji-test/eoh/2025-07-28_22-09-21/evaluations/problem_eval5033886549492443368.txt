def heuristics_v2(df):
    df['avg_daily_return'] = df['close'].pct_change().rolling(window=20).mean()
    df['std_daily_return'] = df['close'].pct_change().rolling(window=20).std()
    df['avg_volume'] = df['volume'].rolling(window=20).mean()
    df['log_volume_diff'] = np.log(df['volume'] / df['avg_volume'])
    df['price_change_pct'] = (df['close'] - df.shift(20)['close']) / df.shift(20)['close']
    heuristics_matrix = (df['avg_daily_return'] / df['std_daily_return'] + df['log_volume_diff']) * df['price_change_pct']
    return heuristics_matrix
