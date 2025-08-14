def heuristics_v2(df):
    df['price_avg_momentum'] = (df['high'].pct_change() + df['low'].pct_change()) / 2
    df['avg_volume'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume_change'] / df['avg_volume']
    df['volatility'] = df['close'].rolling(window=20).std()
    heuristics_matrix = (df['price_avg_momentum'].shift(-1) * df['volume_ratio']) / df['volatility']
    return heuristics_matrix
