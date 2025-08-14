def heuristics_v2(df):
    df['price_momentum'] = df['close'].pct_change(periods=7)
    df['svwap'] = ((df['volume'] * (df['high'] + df['low'] + df['close']) / 3).ewm(span=14).mean()) / (df['volume'].ewm(span=14).mean())
    df['mad'] = df['close'].pct_change().rolling(window=14).apply(lambda x: x.abs().mean(), raw=True)
    heuristics_matrix = (df['price_momentum'].shift(-1) * df['svwap']) / df['mad']
    return heuristics_matrix
