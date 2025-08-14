def heuristics_v2(df):
    df['daily_return'] = df['close'].pct_change()
    df['log_volume'] = np.log(df['volume'])
    df['momentum_indicator'] = (df['daily_return'] * df['log_volume']).rolling(window=60).sum()
    heuristics_matrix = df['momentum_indicator'].dropna()
    return heuristics_matrix
