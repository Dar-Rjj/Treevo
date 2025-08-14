def heuristics_v2(df):
    momentum_window = 14
    cum_volume_window = 5
    log_high_low_window = 20
    
    df['momentum'] = df['close'].pct_change(periods=momentum_window)
    df['cum_volume'] = df['volume'].rolling(window=cum_volume_window).sum()
    df['log_high_low_diff'] = (df['high'] / df['low']).apply(np.log).rolling(window=log_high_low_window).mean()
    
    heuristics_matrix = df['momentum'] * df['cum_volume'] * df['log_high_low_diff']
    return heuristics_matrix
