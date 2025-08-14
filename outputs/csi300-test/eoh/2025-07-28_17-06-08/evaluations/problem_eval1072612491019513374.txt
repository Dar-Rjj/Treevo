def heuristics_v2(df):
    df['mean_price'] = (df['high'].rolling(window=20).mean() + df['low'].rolling(window=20).mean()) / 2
    weights = pd.Series(range(1, 11))
    df['volume_wma'] = df['volume'].rolling(window=10).apply(lambda x: (x * weights).sum() / weights.sum(), raw=True)
    heuristics_matrix = (df['mean_price'] * df['volume_wma']).dropna()
    return heuristics_matrix
