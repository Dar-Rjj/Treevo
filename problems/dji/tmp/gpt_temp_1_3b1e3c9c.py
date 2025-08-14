def heuristics_v2(df):
    close_open_ratio = df['close'] / df['open']
    tr = pd.DataFrame({'h-l': df['high'] - df['low'], 'h-pc': df['high'] - df['close'].shift(1), 'l-pc': df['low'] - df['close'].shift(1)}).max(axis=1)
    atr_14d = tr.rolling(window=14).mean()
    heuristics_matrix = close_open_ratio * atr_14d
    return heuristics_matrix
