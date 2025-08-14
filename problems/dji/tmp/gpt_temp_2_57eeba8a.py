def heuristics_v2(df):
    min_close = df['close'].rolling(window=20).min()
    diff_close_min = df['close'] - min_close
    atr = df[['high', 'low', 'close']].rolling(window=50).apply(lambda x: max(x[0] - x[1], abs(x[0] - x[2]), abs(x[1] - x[2])), raw=True)
    heuristics_matrix = diff_close_min / atr
    return heuristics_matrix
