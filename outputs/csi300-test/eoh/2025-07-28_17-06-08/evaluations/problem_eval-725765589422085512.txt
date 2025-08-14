def heuristics_v2(df):
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['wma_typical_10'] = df['typical_price'].rolling(window=10).apply(lambda x: (x * pd.Series(range(1, len(x) + 1), index=x.index)).sum() / x.sum(), raw=False)
    df['ema_volume_20'] = df['volume'].ewm(span=20, adjust=False).mean()
    df['momentum_5'] = df['close'].pct_change(periods=5)
    heuristics_matrix = (df['wma_typical_10'] * df['ema_volume_20']) * (1 + df['momentum_5'])
    return heuristics_matrix
