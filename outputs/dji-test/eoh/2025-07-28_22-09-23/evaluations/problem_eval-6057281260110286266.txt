def heuristics_v2(df):
    df['daily_return'] = df['close'].pct_change()
    df['mean_30'] = df['close'].rolling(window=30).mean()
    df['relative_diff'] = (df['close'] - df['mean_30']) / df['mean_30']
    df['weighted_diff'] = df['daily_return'] * df['relative_diff']
    df['atr_30'] = df[['high', 'low', 'close']].rolling(window=30).apply(lambda x: np.nanmean([np.max(x) - np.min(x), abs(np.max(x) - x[2]), abs(np.min(x) - x[2])]), raw=False)
    heuristics_matrix = df['weighted_diff'].rolling(window=30).mean() / df['atr_30']
    return heuristics_matrix
