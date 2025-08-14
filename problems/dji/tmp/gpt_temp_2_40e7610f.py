def heuristics_v2(df):
    df['ROC_20'] = (df['close'] - df['close'].shift(20)) / df['close'].shift(20) * 100
    lowest_low = df['low'].rolling(window=14).min()
    highest_high = df['high'].rolling(window=14).max()
    df['Stochastic_14'] = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)
    weights = pd.Series(range(1,51))
    wma_volume_50 = (df['volume'].rolling(window=50).apply(lambda x: (x*weights).sum() / weights.sum(), raw=True))
    ema_close_100 = df['close'].ewm(span=100, adjust=False).mean()
    heuristics_matrix = (df['ROC_20'] * 0.3 + df['Stochastic_14'] * 0.4 + wma_volume_50 * 0.3) * (df['close'] / ema_close_100)
    return heuristics_matrix
