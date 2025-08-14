def heuristics_v2(df):
    highest_30 = df['high'].rolling(window=30).max()
    lowest_10 = df['low'].rolling(window=10).min()
    tr = pd.DataFrame({'h-l':df['high'] - df['low'], 'h-cp':abs(df['high'] - df['close'].shift(1)), 'l-cp':abs(df['low'] - df['close'].shift(1))})
    atr_60 = tr.max(axis=1).rolling(window=60).mean()
    heuristics_matrix = (highest_30 - lowest_10) / atr_60
    return heuristics_matrix
