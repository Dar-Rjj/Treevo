def heuristics_v2(df):
    ten_day_return = df['close'] - df['close'].shift(10)
    atr_50 = df[['high', 'low', 'close']].rolling(window=50).apply(lambda x: np.nanmean([max(x[0] - x[1], x[1] - x[2]), max(x[3] - x[4], x[4] - x[5])]), raw=False)
    heuristics_matrix = ten_day_return / atr_50
    return heuristics_matrix
