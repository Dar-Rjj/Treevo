def heuristics_v2(df):
    high_low_diff = df['high'].rolling(window=20).max() - df['low'].rolling(window=20).min()
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift())
    tr3 = abs(df['low'] - df['close'].shift())
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    avg_true_range = true_range.rolling(window=20).mean()
    heuristics_matrix = high_low_diff / avg_true_range
    return heuristics_matrix
